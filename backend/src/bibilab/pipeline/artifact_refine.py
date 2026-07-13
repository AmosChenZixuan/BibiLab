"""Artifact/mind-map section-batched refine subsystem.

Extracted from worker.py, which stays the SQLite-polling job dispatcher. The
artifact job handler imports the entry points from here: _build_section_views,
_refine_artifact, _refine_mind_map, _render_mind_map_markdown, _MIND_MAP_PROMPT.
Pure move — no signature or behavior changes.
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from bibilab.config import BibilabConfig
from bibilab.db import (
    get_sections,
    get_source,
    get_transcript_segments,
    rows_to_segments,
)
from bibilab.pipeline import _shared as _shared_pipeline
from bibilab.pipeline._shared import (
    _LANG_INSTRUCTION,
    _lang_output_directive,
    _parse_llm_json_response,
    format_mmss,
    resolve_response_language,
)
from bibilab.pipeline.audio import PipelineError
from bibilab.pipeline.transcribe import format_turns

logger = logging.getLogger(__name__)


class ArtifactResult(BaseModel):
    """Result from LLM artifact generation."""

    name: str
    content: str


class MindMapResult(BaseModel):
    """Mind-map LLM output: a single `{name, root}` JSON object. The
    worker renders the markdown file body via `_render_mind_map_markdown`."""

    name: str
    root: dict


# Token-budget knobs for the artifact section-batched refine.
# budget = context_window − 2 * max_output_tokens − _PROMPT_OVERHEAD_TOKENS
# (one max_output for the model's response, one for the running draft fed
# back into batch k>1; the draft is bounded by max_output_tokens.)
_PROMPT_OVERHEAD_TOKENS = 500

# Soft cost note: log a warning when refine batches exceed this threshold.
_SOFT_COST_BATCH_THRESHOLD = 3


@dataclass(frozen=True)
class _SectionView:
    """A section's verbatim text + the metadata needed to render its
    multi-batch header. Built once per source, then greedy-packed into
    batches by the artifact refine flow.

    Invariants:
    - ``text`` is the verbatim output of ``format_turns(include_time=False)``
      over the section's segment slice.
    - ``token_count`` is the section's precomputed DB value and MUST equal
      ``count_tokens(text)`` — both ``_pack_sections`` and the per-section
      budget check trust it.
    - ``timestamp_end >= timestamp_start``.
    - ``source_title`` is a snapshot at build time (intentional: artifact
      output should reflect a stable moment).
    """

    source_id: str
    source_title: str
    seq: int
    timestamp_start: float
    timestamp_end: float
    text: str
    token_count: int


def _pack_sections(
    sections: list[_SectionView],
    budget_tokens: int,
) -> list[list[_SectionView]]:
    """Greedy pack sections into batches respecting the per-batch token budget.

    A section that alone exceeds the budget goes into a batch by itself
    (the caller checks this and raises PipelineError — sections are atomic
    and never split). Order is preserved: section order within each batch
    is the original ``sections`` list order.
    """
    batches: list[list[_SectionView]] = []
    current: list[_SectionView] = []
    current_tokens = 0
    for sec in sections:
        if current and current_tokens + sec.token_count > budget_tokens:
            batches.append(current)
            current = [sec]
            current_tokens = sec.token_count
        else:
            current.append(sec)
            current_tokens += sec.token_count
    if current:
        batches.append(current)
    return batches


def _render_single_batch_text(sections: list[_SectionView]) -> str:
    """Render sections for the single-batch path: byte-identical to today's
    combined-transcript shape.

    Groups by ``source_id`` in input order (which already reflects the
    selected source_ids order — the caller passes them in that order).
    Renders one ``=== Source: {title} ===`` header per source with all its
    sections concatenated, NO per-section header. Sources are joined with
    the same ``"\\n\\n"`` separator today's artifact path uses.

    Per-section text is joined with ``"\\n"`` (NOT ``""``): ``format_turns``
    produces no trailing newline, so a ``""`` join would glue the last
    turn of section N onto the first turn of section N+1 on one mangled
    line. The cost of ``"\\n"`` is a small layout difference from the
    legacy whole-transcript ``format_turns`` (cross-boundary same-speaker
    turns don't merge) — the LLM handles the section-break cleanly.
    """
    by_source: dict[str, list[_SectionView]] = {}
    for sec in sections:
        by_source.setdefault(sec.source_id, []).append(sec)
    parts: list[str] = []
    for source_id, secs in by_source.items():
        title = secs[0].source_title
        text = "\n".join(sec.text for sec in secs)
        parts.append(f"=== Source: {title} ===\n{text}")
    return "\n\n".join(parts)


def _render_multi_batch_section(sec: _SectionView) -> str:
    """Render one section for the multi-batch path: per-section header
    `=== Source: {title} · Section {seq} (mm:ss-mm:ss) ===` followed by the
    section's verbatim text. No trailing newline (the prompt builder
    joins sections)."""
    header = (
        f"=== Source: {sec.source_title} · "
        f"Section {sec.seq} ({format_mmss(sec.timestamp_start)}-"
        f"{format_mmss(sec.timestamp_end)}) ==="
    )
    return f"{header}\n{sec.text}"


# Per-artifact-type refine knobs. The refine loop (`_refine_batched`) is
# shared; only the JSON schema directive shown to the LLM, the multi-batch
# "integrate" wording, and the parse model differ between `artifact` and
# `mind_map`.
_ARTIFACT_SCHEMA_DIRECTIVE = (
    '{\n  "name": "string (a short title for this artifact)",\n'
    '  "content": "string (the main artifact content in markdown format)"\n}'
)
_MIND_MAP_SCHEMA_DIRECTIVE = (
    '{\n  "name": "string (a short title for this mind map)",\n'
    '  "root": "object (recursive tree: {label, evidence, children})"\n}'
)
_ARTIFACT_INTEGRATE_DIRECTIVE = (
    "Integrate this new material into the draft. Keep the same JSON "
    "schema; refine the draft's name and content to reflect the "
    "accumulated material."
)
_MIND_MAP_INTEGRATE_DIRECTIVE = (
    "Integrate this new material into the draft. Refine the draft's "
    "name and root tree to reflect the accumulated material. Keep each "
    "node's existing 'evidence' quote verbatim (copy it unchanged); only "
    "replace a quote if you are pulling a fresh verbatim passage from the "
    "new material for that node."
)


def _build_initial_prompt(
    prompt: str,
    transcript_text: str,
    cfg: BibilabConfig,
    ui_lang: str | None,
    schema_directive: str = _ARTIFACT_SCHEMA_DIRECTIVE,
) -> str:
    """Build the prompt for the first batch of a section-batched refine,
    or for the single-call path when everything fits in one batch.

    The artifact single-call case is byte-identical to today's
    _generate_artifact template (regression guard). `schema_directive` is
    the JSON shape the LLM must return — defaults to ArtifactResult's
    {name, content}; mind_map passes {name, root}."""
    lang = resolve_response_language(cfg.ai, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    lang_output_directive = _lang_output_directive(lang)
    return f"""{lang_instruction}

{prompt}

Based on the following transcripts, generate the requested artifact content.

Transcript:
{transcript_text}

{lang_instruction}
Respond ONLY with valid JSON matching this schema:
{schema_directive}
{lang_output_directive}"""


def _build_refine_prompt(
    prompt: str,
    draft: BaseModel,
    new_sections_text: str,
    cfg: BibilabConfig,
    ui_lang: str | None,
    schema_directive: str = _ARTIFACT_SCHEMA_DIRECTIVE,
    draft_label: str = "name + content",
    integrate_directive: str = _ARTIFACT_INTEGRATE_DIRECTIVE,
) -> str:
    """Build the prompt for batch k>1: show the running draft, show the
    new material, and instruct the LLM to integrate them. The LLM returns
    a fresh JSON object matching `schema_directive`. Defaults are
    ArtifactResult's {name, content}; mind_map overrides the three knobs."""
    lang = resolve_response_language(cfg.ai, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    lang_output_directive = _lang_output_directive(lang)
    # Use json.dumps (not repr interpolation) so the fenced JSON block is
    # actually valid JSON for the LLM to read. ensure_ascii=False keeps
    # non-ASCII characters readable for non-English outputs.
    draft_block = json.dumps(draft.model_dump(), ensure_ascii=False)
    draft_text = f"Current draft ({draft_label}):\n```json\n{draft_block}\n```\n"
    return f"""{lang_instruction}

{prompt}

{draft_text}

New material to integrate:
{new_sections_text}

{integrate_directive}

{lang_instruction}
Respond ONLY with valid JSON matching this schema:
{schema_directive}
{lang_output_directive}"""


# Mind-map artifact (type='mind_map'). The LLM emits a single JSON object
# matching `MindMapResult` ({name, root}) — no `content` envelope, no fenced
# block. It shares the `_refine_batched` loop via `_MIND_MAP_SPEC`.
_MIND_MAP_PROMPT = """\
Produce a hierarchical mind map that captures the central topic and its
sub-themes across the supplied transcript(s).

Respond with a single JSON object (no surrounding markdown, no fenced
code block) matching this exact shape:

  {
    "name": "string (a short title for this mind map)",
    "root": {
      "label": "string (root node, 2-6 words)",
      "evidence": "string (a verbatim quote, see rule 3)",
      "children": [
        {"label": "string (branch)", "evidence": "string",
         "children": [{"label": "string", "evidence": "string"}, ...]},
        ...
      ]
    }
  }

Rules:

1. Hierarchy (cap total nodes at ~30; do not pad):
   - Root: the overall topic.
   - 2-5 main branches (level 1).
   - 1-5 children per branch (level 2+).

2. Node label rules:
   - Keep labels SHORT: a single phrase, ideally 1-6 words.
   - Mirror the majority language of the source transcript.
   - Do NOT use quotation marks, backslashes, or unescaped newlines
     inside labels.
   - Do NOT use markdown formatting (bold, italic, links) inside labels.

3. Node "evidence" rules:
   - For EVERY node, include an "evidence" field: one short verbatim
     quote (1-2 sentences) copied from a source transcript exactly as
     written — no paraphrase. A node label is often a synthesized
     abstraction; the quote is the literal passage that grounds it, so a
     later search can find the source.
   - Quote in the transcript's own language even when the label is in
     another language.
   - Prefer a passage naming the node's topic, entities, or events over
     generic framing.
   - Escape any quotation marks, backslashes, or newlines so the quote is
     valid inside the JSON string; if a passage cannot be escaped cleanly,
     pick a different one.

4. No explanatory text outside the JSON object."""


def _render_mind_map_markdown(mm: MindMapResult) -> str:
    """Build the artifact file body for a mind_map job: one ```json fence
    holding `{"root": ...}`."""
    payload = json.dumps({"root": mm.root}, ensure_ascii=False, indent=2)
    return f"```json\n{payload}\n```\n"


async def _build_section_views(source_ids: list[str]) -> list[_SectionView]:
    """Load each source's sections, reconstruct their verbatim text from
    the segment slice, and return a flat ordered list of _SectionView
    (selected-source order, then seq within source).

    A source with no section rows raises ``PipelineError("...no sections;
    re-ingest required")`` (the contract: every source that flows through
    the artifact pipeline has section rows). Segments are loaded once per
    source; the text is ``format_turns`` with ``include_time=False`` (matches
    the pre-batching combined-transcript shape).
    """
    views: list[_SectionView] = []
    for source_id in source_ids:
        source = await get_source(source_id)
        if source is None:
            raise PipelineError(f"Source {source_id!r} not found")
        rows = await get_sections(source_id)
        if not rows:
            raise PipelineError(f"Source {source_id!r} has no sections; re-ingest required")
        seg_rows = await get_transcript_segments(source_id)
        if not seg_rows:
            raise PipelineError(f"Source {source_id!r} has no transcript")
        by_seq = {r["seq"]: r for r in seg_rows}
        title = source["title"]
        for row in rows:
            slice_rows = [by_seq[s] for s in range(row["seg_start"], row["seg_end"] + 1) if s in by_seq]
            if not slice_rows:
                # The atomic write contract (sections + segments land in
                # the same transaction) makes this unreachable in production,
                # but fail-loud rather than silently render an empty section.
                raise PipelineError(
                    f"Source {source_id!r} section {row['seq']} references "
                    f"missing segments ({row['seg_start']}..{row['seg_end']}); "
                    f"re-ingest required"
                )
            slice_segs = rows_to_segments(slice_rows)
            text = format_turns(slice_segs, include_time=False)
            views.append(
                _SectionView(
                    source_id=source_id,
                    source_title=title,
                    seq=row["seq"],
                    timestamp_start=row["timestamp_start"],
                    timestamp_end=row["timestamp_end"],
                    text=text,
                    token_count=row["token_count"],
                )
            )
    return views


@dataclass(frozen=True)
class _RefineSpec:
    """The per-artifact-type knobs `_refine_batched` varies over: the
    Pydantic model that parses the LLM response, the JSON schema directive
    and multi-batch "integrate" wording shown to the LLM, the running-draft
    label, and the batch-label prefix used in retry logs / error messages."""

    model: type[BaseModel]
    schema_directive: str
    draft_label: str
    integrate_directive: str
    label_prefix: str


_ARTIFACT_SPEC = _RefineSpec(
    model=ArtifactResult,
    schema_directive=_ARTIFACT_SCHEMA_DIRECTIVE,
    draft_label="name + content",
    integrate_directive=_ARTIFACT_INTEGRATE_DIRECTIVE,
    label_prefix="artifact",
)
_MIND_MAP_SPEC = _RefineSpec(
    model=MindMapResult,
    schema_directive=_MIND_MAP_SCHEMA_DIRECTIVE,
    draft_label="name + root",
    integrate_directive=_MIND_MAP_INTEGRATE_DIRECTIVE,
    label_prefix="mind_map",
)


async def _refine_batched(
    *,
    prompt: str,
    sections: list[_SectionView],
    cfg: BibilabConfig,
    ui_lang: str | None,
    spec: _RefineSpec,
) -> BaseModel:
    """Section-batched, running-draft refine shared by all artifact types.

    When all sections fit in one batch (the common case), this collapses to
    a single ``_call_llm`` with the ``_build_initial_prompt`` template (the
    artifact single-batch path stays byte-identical to the legacy template).
    When they don't fit, batch 1 produces the initial draft and batch k>1
    refines the running draft with new sections.

    A single section that alone exceeds the per-batch budget raises
    ``PipelineError`` (sections are atomic — never split); a batch whose
    retry ladder is exhausted also raises (no partial artifact). Batch count
    over ``_SOFT_COST_BATCH_THRESHOLD`` logs a soft cost note. The only
    per-type variation is carried by ``spec``.
    """
    budget = cfg.ai.context_window - 2 * cfg.ai.max_output_tokens - _PROMPT_OVERHEAD_TOKENS
    if budget <= 0:
        raise PipelineError(
            f"{spec.label_prefix} batch budget non-positive (context_window={cfg.ai.context_window}, "
            f"2*max_output_tokens={2 * cfg.ai.max_output_tokens}, "
            f"prompt_overhead={_PROMPT_OVERHEAD_TOKENS}); "
            f"raise max_output_tokens or context_window"
        )

    # Atomicity guard: a section alone > budget cannot be split. The packing
    # is greedy, so an oversized section is the only thing that can produce a
    # batch over budget. Fail loud here, before the LLM call.
    for sec in sections:
        if sec.token_count > budget:
            raise PipelineError(
                f"Source {sec.source_id!r} section {sec.seq} alone "
                f"({sec.token_count} tokens) exceeds the {spec.label_prefix} batch "
                f"budget ({budget} tokens); reduce section size or raise "
                f"context_window"
            )

    batches = _pack_sections(sections, budget)
    if len(batches) > _SOFT_COST_BATCH_THRESHOLD:
        logger.warning(
            "%s refine using %d batches (threshold=%d); consider raising context_window or selecting fewer sources",
            spec.label_prefix,
            len(batches),
            _SOFT_COST_BATCH_THRESHOLD,
        )

    draft: BaseModel | None = None
    for i, batch in enumerate(batches, start=1):
        label = f"{spec.label_prefix} batch {i}/{len(batches)}"
        if i == 1:
            # Single-batch renders ALL sections (byte-identical guard);
            # multi-batch batch 1 renders only its own sections.
            if len(batches) == 1:
                text = _render_single_batch_text(sections)
            else:
                text = "\n".join(_render_multi_batch_section(s) for s in batch)
            llm_prompt = _build_initial_prompt(
                prompt,
                text,
                cfg,
                ui_lang,
                schema_directive=spec.schema_directive,
            )
        else:
            assert draft is not None
            text = "\n".join(_render_multi_batch_section(s) for s in batch)
            llm_prompt = _build_refine_prompt(
                prompt,
                draft,
                text,
                cfg,
                ui_lang,
                schema_directive=spec.schema_directive,
                draft_label=spec.draft_label,
                integrate_directive=spec.integrate_directive,
            )
        draft = await asyncio.to_thread(
            _shared_pipeline._call_llm_with_retry,
            [llm_prompt],
            lambda raw: _parse_llm_json_response(raw, spec.model),
            cfg=cfg.ai,
            label=label,
            max_attempts=3,
        )
    assert draft is not None
    return draft


async def _refine_artifact(
    *,
    prompt: str,
    sections: list[_SectionView],
    cfg: BibilabConfig,
    ui_lang: str | None = None,
) -> ArtifactResult:
    """Section-batched refine for text artifacts (brief, study guide, …)."""
    result = await _refine_batched(prompt=prompt, sections=sections, cfg=cfg, ui_lang=ui_lang, spec=_ARTIFACT_SPEC)
    assert isinstance(result, ArtifactResult)
    return result


async def _refine_mind_map(
    *,
    sections: list[_SectionView],
    cfg: BibilabConfig,
    ui_lang: str | None = None,
) -> MindMapResult:
    """Section-batched refine for mind_map: same loop as `_refine_artifact`,
    but the LLM is asked for `{name, root}` and the result parses as
    MindMapResult. The worker renders the file body via
    `_render_mind_map_markdown`."""
    result = await _refine_batched(
        prompt=_MIND_MAP_PROMPT, sections=sections, cfg=cfg, ui_lang=ui_lang, spec=_MIND_MAP_SPEC
    )
    assert isinstance(result, MindMapResult)
    return result
