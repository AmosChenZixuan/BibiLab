"""Background worker loop."""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from bibilab.adapters.base import AuthRequiredError, VideoMeta
from bibilab.cleanup import cleanup_job_artifacts, purge_download_files
from bibilab.config import BibilabConfig, bibilab_home, load_config
from bibilab.db import (
    apply_digest_facets,
    create_artifact,
    delete_job,
    get_list,
    get_pending_jobs,
    get_sections,
    get_source,
    get_transcript_segments,
    parse_job_meta,
    rows_to_sections,
    rows_to_segments,
    update_job_meta,
    update_job_status,
    update_section_summaries,
    write_source_with_segments,
)
from bibilab.model_registry import ensure
from bibilab.models.jobs import JobStatus

# Aliased module import for _call_llm_with_retry (long signature; kept off the explicit
# import list to avoid dragging every kwarg onto a bare-name import line).
from bibilab.pipeline import _shared as _shared_pipeline
from bibilab.pipeline._shared import (
    _LANG_INSTRUCTION,
    _lang_output_directive,
    _parse_llm_json_response,
    resolve_response_language,
)
from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.digest import DigestResult, SectionDigest, digest_sections
from bibilab.pipeline.embed import embed_chunks
from bibilab.pipeline.punctuate import punctuate
from bibilab.pipeline.section import Section, chunk_by_sections, derive_sections, section_texts
from bibilab.pipeline.transcribe import (
    WhisperSegment,
    format_turns,
    transcribe,
)

logger = logging.getLogger(__name__)


async def reset_stuck_jobs() -> None:
    from bibilab.db import _in_placeholders, _now, get_db
    from bibilab.models.jobs import ACTIVE_JOB_STATUSES, JobStatus

    stuck = tuple(s for s in ACTIVE_JOB_STATUSES if s is not JobStatus.QUEUED)
    placeholders = _in_placeholders(list(stuck))
    async with get_db() as db:
        await db.execute(
            f"""
            UPDATE jobs SET status=?, updated_at=?
            WHERE status IN ({placeholders})
            """,
            (JobStatus.QUEUED, _now(), *stuck),
        )
        await db.commit()


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


def _format_duration(seconds: float) -> str:
    """mm:ss formatter for section headers. Past 1h, keeps counting minutes
    (no H:MM:SS) — the header is for orientation, not parsing."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


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
        f"Section {sec.seq} ({_format_duration(sec.timestamp_start)}-"
        f"{_format_duration(sec.timestamp_end)}) ==="
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


def _download_cover(cover_url: str, dest: Path) -> bool:
    """Download cover image from URL to dest path. Returns True on success."""
    try:
        resp = httpx.get(cover_url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Job: failed to download cover from %s: %s", cover_url, exc)
        return False


class WorkerLoop:
    def __init__(
        self,
        concurrency: int = 1,
        config: BibilabConfig | None = None,
        adapter: Any = None,
        home: Path | None = None,
    ) -> None:
        self._config = config
        self._adapter = adapter
        self._bibilab_home = home if home is not None else bibilab_home()
        self._concurrency = concurrency
        self._task: asyncio.Task | None = None
        self._running = False
        self._cancelled: set[str] = set()
        self._in_flight: set[str] = set()

    async def start(self) -> None:
        await reset_stuck_jobs()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="bibilab-worker")
        logger.info("Worker loop started (concurrency=%d)", self._concurrency)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Worker loop stopped")

    def cancel_job(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    def _get_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        cfg = self._get_config()
        from bibilab.adapters.bilibili import BilibiliAdapter

        return BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie)

    def _get_config(self) -> BibilabConfig:
        if self._config is not None:
            return self._config
        return load_config()

    async def _loop(self) -> None:
        while self._running:
            if len(self._in_flight) < self._concurrency:
                pending = [dict(j) for j in await get_pending_jobs()]
                queued = [
                    j for j in pending if j["status"] == JobStatus.QUEUED.value and j["id"] not in self._in_flight
                ]
                slots = self._concurrency - len(self._in_flight)
                for job in queued[:slots]:
                    self._in_flight.add(job["id"])
                    asyncio.create_task(self._run_job(job))
            await asyncio.sleep(2)

    async def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        try:
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            if job["type"] == "artifact":
                await self._run_artifact_job(job)
                return

            if job["type"] == "model_download":
                await self._download_model_job(job)
                return

            if job["type"] == "digest":
                await self._run_digest_job(job)
                return

            await self._pipeline(job)

        except AuthRequiredError as exc:
            logger.warning("Job %s needs auth (%s)", job_id, exc.resource_type)
            await update_job_status(job_id, JobStatus.NEEDS_AUTH.value, error=exc.resource_type)
        except asyncio.CancelledError:
            # Re-raise CancelledError - don't swallow it as a job failure
            raise
        except PipelineError as exc:
            logger.exception("Job %s pipeline error: %s", job_id, exc)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
        except Exception as exc:
            logger.exception("Job %s (type=%s) failed", job_id, job.get("type", "unknown"))
            if job.get("type") in (None, "ingest"):
                await asyncio.to_thread(cleanup_job_artifacts, job)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
        finally:
            self._in_flight.discard(job_id)
            self._cancelled.discard(job_id)

    async def _download_model_job(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = parse_job_meta(job)
        spec_id = meta_raw.get("model_name", "")

        if not spec_id:
            logger.error("Model download job %s missing model_name in meta", job_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error="missing model_name in job meta")
            return

        await update_job_status(job_id, JobStatus.DOWNLOADING.value, progress=10)
        await asyncio.to_thread(ensure, spec_id)
        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Model download job %s completed for %s", job_id, spec_id)

    # -------------------------------------------------------------------------
    # Artifact generation job
    # -------------------------------------------------------------------------

    async def _run_artifact_job(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = parse_job_meta(job)
        artifact_id = meta_raw["artifact_id"]
        list_id = meta_raw["list_id"]
        artifact_type = meta_raw["type"]
        prompt = meta_raw["prompt"]
        source_ids = meta_raw["source_ids"]
        cfg = self._get_config()
        # Mind-map jobs ignore the user-supplied prompt and use
        # `_MIND_MAP_PROMPT` instead. The rebind also feeds
        # `create_artifact`'s `prompt` column so the view-prompt modal
        # shows what the LLM actually saw.
        is_mind_map = artifact_type == "mind_map"
        if is_mind_map:
            prompt = _MIND_MAP_PROMPT

        try:
            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=10)

            # Cancellation check before LLM work.
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=30)

            # _build_section_views handles source existence + no-sections
            # fail-loud.
            sections = await _build_section_views(source_ids)
            if is_mind_map:
                mind_map_result = await _refine_mind_map(
                    sections=sections,
                    cfg=cfg,
                    ui_lang=meta_raw.get("ui_lang"),
                )
                content = _render_mind_map_markdown(mind_map_result)
                artifact_name = mind_map_result.name
            else:
                artifact_result = await _refine_artifact(
                    prompt=prompt,
                    sections=sections,
                    cfg=cfg,
                    ui_lang=meta_raw.get("ui_lang"),
                )
                content = artifact_result.content
                artifact_name = artifact_result.name

            # Check for cancellation before writing file
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=80)

            # Write artifact content to file
            artifacts_dir = self._bibilab_home / "artifacts" / list_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            content_path = artifacts_dir / f"{artifact_id}.md"
            content_path.write_text(content, encoding="utf-8")

            # Create artifact record with success status
            await create_artifact(
                artifact_id=artifact_id,
                list_id=list_id,
                name=artifact_name,
                type=artifact_type,
                prompt=prompt,
                source_ids=source_ids,
                status="completed",
                content_path=str(content_path.relative_to(self._bibilab_home)),
                error=None,
            )

            await update_job_status(job_id, JobStatus.DONE.value, progress=100)
            logger.info("Artifact job %s completed for artifact %s", job_id, artifact_id)

        except PipelineError as exc:
            error_message = str(exc)
            logger.error("Artifact job %s failed: %s", job_id, error_message)
            await update_job_status(job_id, JobStatus.FAILED.value, error=error_message)

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    async def _call_digest_sections(
        self,
        section_texts_list: list[str],
        video_meta: VideoMeta,
        cfg: BibilabConfig,
        ui_lang: str | None = None,
    ) -> tuple[DigestResult, list[SectionDigest]]:
        """Run the section-level digest. Does not catch exceptions — callers handle errors."""
        return await asyncio.to_thread(
            digest_sections,
            section_texts_list,
            video_meta,
            cfg.ai,
            cfg.ai.output_language,
            ui_lang,
        )

    # -------------------------------------------------------------------------
    # Digest-only job (rerun)
    # -------------------------------------------------------------------------

    async def _run_digest_job(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = parse_job_meta(job)
        source_id: str = meta_raw.get("source_id", "")
        cfg = self._get_config()

        source = await get_source(source_id)
        if source is None:
            logger.warning("Digest job %s: source %s not found", job_id, source_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error=f"Source {source_id!r} not found")
            return

        segments = await get_transcript_segments(source_id)
        if not segments:
            logger.warning("Digest job %s: source %s has no transcript", job_id, source_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error="Source has no transcript")
            return

        whisper_segments = rows_to_segments(segments)

        section_rows = await get_sections(source_id)
        if not section_rows:
            # Defensive: unreachable through the normal ingest path (sections
            # are written atomically with the source). Surface a re-ingest hint
            # if it ever happens.
            msg = "Source has no sections; re-ingest the source to derive them"
            logger.error("Digest job %s: %s", job_id, msg)
            await update_job_status(job_id, JobStatus.FAILED.value, error=msg)
            return

        sections = rows_to_sections(section_rows)
        section_texts_list = section_texts(whisper_segments, sections)
        video_meta = VideoMeta.from_source(source)

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=10)

        try:
            extraction, section_digests = await self._call_digest_sections(
                section_texts_list, video_meta, cfg, meta_raw.get("ui_lang")
            )
        except Exception as exc:
            logger.exception("Digest job %s failed", job_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
            return

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=80)

        # Two writes: per-section summaries, then the source's facet columns.
        # No transaction needed — they touch disjoint rows/columns and a
        # rerun is idempotent (re-running fixes any partial write).
        await update_section_summaries(
            source_id,
            [(row["seq"], sd.summary, sd.keywords) for row, sd in zip(section_rows, section_digests)],
        )
        await apply_digest_facets(
            source_id,
            series_name=extraction.series_name,
            sequence_number=extraction.sequence_number,
            season_number=extraction.season_number,
            bump_processed_at=False,
        )

        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Digest job %s completed for source %s", job_id, source_id)

    # -------------------------------------------------------------------------
    # Pipeline stages (full ingest)
    # -------------------------------------------------------------------------

    async def _pipeline(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = parse_job_meta(job)
        cfg = self._get_config()

        source_id: str = meta_raw.get("source_id") or str(uuid.uuid4())
        meta_raw["source_id"] = source_id
        job["meta"] = meta_raw  # cleanup snapshot must carry source_id (Q1)
        await update_job_meta(job_id, {"source_id": source_id})
        video_id: str = meta_raw["video_id"]
        list_id: str = meta_raw["list_id"]

        list_row = await get_list(list_id)
        if list_row is None:
            raise PipelineError(f"List {list_id!r} not found - it may have been deleted")

        video_meta = VideoMeta(
            video_id=video_id,
            title=meta_raw.get("title", ""),
            platform=meta_raw.get("platform", ""),
            source_url=meta_raw.get("source_url", ""),
            cover_url=meta_raw.get("cover_url", ""),
            duration_seconds=meta_raw.get("duration_seconds", 0),
            uploader=meta_raw.get("uploader", ""),
        )

        # Stage 1: Download
        try:
            video_path = await self._stage_download(job_id, video_meta, source_id)
        except Exception as exc:
            raise PipelineError(f"[downloading] {exc}") from exc
        if video_path is None:
            return  # cancelled

        # Stage 2: Audio extraction
        try:
            wav_path = await self._stage_extract_audio(job_id, video_path, video_meta.duration_seconds)
        except Exception as exc:
            raise PipelineError(f"[transcribing] {exc}") from exc
        if wav_path is None:
            return  # cancelled

        # Stage 3: Transcription
        try:
            result = await self._stage_transcribe(job_id, wav_path, source_id, cfg)
        except Exception as exc:
            raise PipelineError(f"[transcribing] {exc}") from exc
        if result is None:
            return  # cancelled
        detected_language, effective_language, sentence_segments = result

        # Stage 4: Derive sections, chunk per-section, digest + embed in parallel
        try:
            result = await self._stage_process(
                job_id, job, sentence_segments, source_id, video_meta, list_id, cfg, effective_language
            )
        except Exception as exc:
            raise PipelineError(f"[processing] {exc}") from exc
        if result is None:
            return  # cancelled
        extraction, sections, section_digests = result

        # Stage 5: Persist source + segments + sections atomically, then cleanup
        try:
            await self._stage_persist(
                job_id,
                source_id,
                video_id,
                video_meta,
                list_id,
                extraction,
                sections,
                section_digests,
                detected_language,
                cfg,
                sentence_segments,
            )
        except Exception as exc:
            raise PipelineError(f"[persisting] {exc}") from exc
        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Job %s completed for video %s", job_id, video_id)

    async def _abort_if_cancelled(self, job_id: str) -> bool:
        """If the job is cancelled, purge its artifacts, delete it, and report
        True so the caller stops. Safe to call more than once within a stage."""
        if job_id not in self._cancelled:
            return False
        self._cancelled.discard(job_id)
        await asyncio.to_thread(cleanup_job_artifacts, {"id": job_id})
        await delete_job(job_id)
        return True

    async def _stage_download(
        self,
        job_id: str,
        video_meta: VideoMeta,
        source_id: str,
    ) -> Path | None:
        """Stage 1: Download video file and cover image."""
        await update_job_status(job_id, JobStatus.DOWNLOADING.value, progress=10)
        if await self._abort_if_cancelled(job_id):
            return None

        # .part hygiene: purge any downloads/{id}.* left by a prior failed/corrupt
        # attempt so this download starts clean and never resumes onto stale bytes.
        await asyncio.to_thread(purge_download_files, video_meta.video_id)

        if await self._abort_if_cancelled(job_id):
            return None
        video_path: Path = await asyncio.to_thread(
            self._get_adapter().download,
            video_meta.video_id,
            video_meta.source_url,
            self._get_config().backend.download_connections,
        )

        # Download cover
        covers_dir = self._bibilab_home / "covers"
        covers_dir.mkdir(parents=True, exist_ok=True)
        cover_dest = covers_dir / f"{source_id}.jpg"
        await asyncio.to_thread(_download_cover, video_meta.cover_url, cover_dest)

        return video_path

    async def _stage_extract_audio(
        self,
        job_id: str,
        video_path: Path,
        expected_duration: float = 0.0,
    ) -> Path | None:
        """Stage 2: Extract audio from downloaded video."""
        await update_job_status(job_id, JobStatus.TRANSCRIBING.value, progress=25)
        try:
            wav_path = await asyncio.to_thread(extract_audio, video_path, expected_duration)
        except Exception:
            # video_path exists but audio extraction failed - video will be cleaned up
            # by cancel_or_delete_job using the full job dict from DB
            raise

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, {"id": job_id})
            await delete_job(job_id)
            return None

        return wav_path

    async def _stage_transcribe(
        self,
        job_id: str,
        wav_path: Path,
        source_id: str,
        cfg: BibilabConfig,
    ) -> tuple | None:
        """Stage 3: Transcribe audio, punctuate (zh-gated) into sentence segments."""
        await update_job_status(job_id, JobStatus.TRANSCRIBING.value, progress=30)
        vad_segments, detected_language = await asyncio.to_thread(transcribe, wav_path, cfg.transcription)
        wav_path.unlink(missing_ok=True)  # clean up early — punctuate only needs text
        effective_language = (
            cfg.transcription.language if cfg.transcription.language != "auto" else (detected_language or "en")
        )
        sentence_segments = await asyncio.to_thread(punctuate, vad_segments, effective_language)

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, {"id": job_id})
            await delete_job(job_id)
            return None

        return detected_language, effective_language, sentence_segments

    async def _stage_process(
        self,
        job_id: str,
        job: dict,
        sentence_segments: list[WhisperSegment],
        source_id: str,
        video_meta: VideoMeta,
        list_id: str,
        cfg: BibilabConfig,
        effective_language: str,
    ) -> tuple[DigestResult, list[Section], list[SectionDigest]] | None:
        """Stage 4: Derive sections, chunk per-section, run digest + embed in parallel.

        Returns (extraction, sections, section_digests). The 1-section case
        produces a 1-element section_digests mirroring the DigestResult.

        Pipeline order: `segments → sections → chunks` (section-first, chunk-within).
        Chunks are produced from per-section slices, so a chunk can never cross a
        section boundary — the `chunk → section → source` nesting is physical, not
        by convention. Re-stamping in `chunk_by_sections` keeps the chunks' seg
        indices and sequence_index source-global so the rest of the system is
        unaware sections exist (the per-chunk section FK is a future
        optimization; today the chunk→section containment is a structural
        property of chunk_by_sections, not a stored relationship).
        """
        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=40)
        sections = derive_sections(sentence_segments)
        chunks = chunk_by_sections(sentence_segments, sections, language=effective_language)
        section_texts_list = section_texts(sentence_segments, sections)

        meta_raw = parse_job_meta(job)
        ui_lang = meta_raw.get("ui_lang")

        async def _digest() -> tuple[DigestResult, list[SectionDigest]]:
            return await self._call_digest_sections(section_texts_list, video_meta, cfg, ui_lang)

        async def _embed() -> None:
            await asyncio.to_thread(embed_chunks, chunks, source_id, video_meta, list_id)

        gather_results = await asyncio.gather(_digest(), _embed(), return_exceptions=True)
        digest_raw, embed_raw = gather_results
        if isinstance(embed_raw, BaseException):
            logger.error("embed_chunks raised but was not the primary error", exc_info=embed_raw)
        if isinstance(digest_raw, BaseException):
            raise digest_raw
        if isinstance(embed_raw, BaseException):
            raise embed_raw
        extraction, section_digests = digest_raw

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await delete_job(job_id)
            return None

        return extraction, sections, section_digests

    async def _stage_persist(
        self,
        job_id: str,
        source_id: str,
        video_id: str,
        video_meta: VideoMeta,
        list_id: str,
        extraction: DigestResult,
        sections: list[Section],
        section_digests: list[SectionDigest],
        detected_language: str,
        cfg: BibilabConfig,
        sentence_segments: list[WhisperSegment],
    ) -> None:
        """Stage 5: Persist source row + transcript segments + section rows
        atomically, then cleanup. All three land in one transaction (or none);
        a partial write rolls back so re-ingest never leaves orphan rows."""
        await write_source_with_segments(
            segments=sentence_segments,
            sections=sections,
            section_digests=section_digests,
            source_id=source_id,
            video_id=video_id,
            platform=video_meta.platform,
            list_id=list_id,
            title=video_meta.title,
            cover_url=video_meta.cover_url,
            source_url=video_meta.source_url,
            duration_seconds=video_meta.duration_seconds,
            uploader=video_meta.uploader,
            language=detected_language,
            whisper_model=cfg.transcription.model,
            ai_model=cfg.ai.model,
            settings_snapshot=cfg.model_dump(),
            series_name=extraction.series_name,
            sequence_number=extraction.sequence_number,
            season_number=extraction.season_number,
        )

        # Cleanup downloads
        purge_download_files(video_id)
