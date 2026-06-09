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
from bibilab.cleanup import cleanup_job_artifacts
from bibilab.config import BibilabConfig, bibilab_home, load_config
from bibilab.db import (
    create_artifact,
    delete_job,
    get_list,
    get_pending_jobs,
    get_section_ranges,
    get_source,
    get_transcript_segments,
    parse_job_meta,
    reset_stuck_jobs,
    update_job_meta,
    update_job_status,
    update_source_digest,
    write_source_with_segments,
)
from bibilab.model_registry import ensure
from bibilab.models.jobs import JobStatus
from bibilab.pipeline._shared import (
    _LANG_INSTRUCTION,
    ContextWindowExceededError,
    _call_llm,
    _lang_output_directive,
    _parse_llm_json_response,
    _resolved_lang,
)
from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.digest import DigestResult, digest
from bibilab.pipeline.embed import embed_chunks
from bibilab.pipeline.punctuate import punctuate
from bibilab.pipeline.section import Section, chunk_by_sections, derive_sections
from bibilab.pipeline.transcribe import (
    WhisperSegment,
    _row_to_whisper_segment,
    format_turns,
    load_transcript_text,
    transcribe,
)

logger = logging.getLogger(__name__)


class ArtifactResult(BaseModel):
    """Result from LLM artifact generation."""

    name: str
    content: str


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


def _build_initial_prompt(
    prompt: str,
    transcript_text: str,
    cfg: BibilabConfig,
    ui_lang: str | None,
) -> str:
    """Build the prompt for the first batch of a section-batched refine,
    or for the single-call path when everything fits in one batch.

    The single-call case is byte-identical to today's _generate_artifact
    template (regression guard)."""
    lang = _resolved_lang(cfg.ai.output_language, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    lang_output_directive = _lang_output_directive(lang)
    return f"""{lang_instruction}

{prompt}

Based on the following transcripts, generate the requested artifact content.

Transcript:
{transcript_text}

{lang_instruction}
Respond ONLY with valid JSON matching this schema:
{{
  "name": "string (a short title for this artifact)",
  "content": "string (the main artifact content in markdown format)"
}}
{lang_output_directive}"""


def _build_refine_prompt(
    prompt: str,
    draft: "ArtifactResult",
    new_sections_text: str,
    cfg: BibilabConfig,
    ui_lang: str | None,
) -> str:
    """Build the prompt for batch k>1: show the running draft, show the
    new material, and instruct the LLM to integrate them. The LLM returns
    a fresh {name, content} JSON — name is re-derived from accumulated
    context (single prompt template family, no special-case)."""
    lang = _resolved_lang(cfg.ai.output_language, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    lang_output_directive = _lang_output_directive(lang)
    # Use json.dumps (not repr interpolation) so the fenced JSON block is
    # actually valid JSON for the LLM to read. ensure_ascii=False keeps
    # non-ASCII characters readable for non-English outputs.
    draft_block = json.dumps({"name": draft.name, "content": draft.content}, ensure_ascii=False)
    draft_text = f"Current draft (name + content):\n```json\n{draft_block}\n```\n"
    integrate_directive = (
        "Integrate this new material into the draft. Keep the same JSON "
        "schema; refine the draft's name and content to reflect the "
        "accumulated material."
    )
    return f"""{lang_instruction}

{prompt}

{draft_text}

New material to integrate:
{new_sections_text}

{integrate_directive}

{lang_instruction}
Respond ONLY with valid JSON matching this schema:
{{
  "name": "string (a short title for this artifact)",
  "content": "string (the main artifact content in markdown format)"
}}
{lang_output_directive}"""


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
        rows = await get_section_ranges(source_id)
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
            slice_segs = [_row_to_whisper_segment(r) for r in slice_rows]
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


async def _call_llm_with_retry(
    llm_prompt: str,
    cfg: BibilabConfig,
    *,
    label: str,
) -> ArtifactResult:
    """Run ``_call_llm`` with the standard 3-attempt retry ladder. On
    ``ContextWindowExceededError``, raise immediately (deterministic — the
    same prompt would re-overflow). On other failures, retry 3 times then
    raise ``PipelineError``."""
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = await asyncio.to_thread(_call_llm, llm_prompt, cfg.ai)
            return _parse_llm_json_response(raw, ArtifactResult)
        except ContextWindowExceededError:
            # Deterministic — no point retrying.
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("LLM %s failed (attempt %d/3): %s", label, attempt + 1, exc)
            if attempt < 2:
                continue
    error_msg = f"LLM {label} exhausted all retries: {last_exc}"
    logger.error(error_msg)
    raise PipelineError(error_msg)


async def _refine_artifact(
    *,
    prompt: str,
    sections: list[_SectionView],
    cfg: BibilabConfig,
    ui_lang: str | None = None,
) -> ArtifactResult:
    """Section-batched, running-draft refine for the artifact pipeline.

    When all sections fit in one batch (the common case — a few short
    sources, each with 1 section), this collapses to the legacy
    single-``_call_llm`` behavior, with the byte-identical prompt template
    produced by ``_build_initial_prompt``.

    When sections don't fit, the multi-batch path calls ``_call_llm`` once
    per batch: batch 1 produces an initial draft; batch k>1 refines the
    running draft with new sections.

    A single section that alone exceeds the per-batch budget raises
    ``PipelineError`` (sections are atomic — never split). A batch whose
    ``_call_llm`` exhausts the 3-attempt retry ladder raises
    ``PipelineError`` (no partial artifact).

    When the batch count exceeds ``_SOFT_COST_BATCH_THRESHOLD``, a soft
    cost note is logged via ``logger.warning`` (no schema/UI change).
    """
    budget = cfg.ai.context_window - 2 * cfg.ai.max_output_tokens - _PROMPT_OVERHEAD_TOKENS
    if budget <= 0:
        raise PipelineError(
            f"Artifact batch budget non-positive (context_window={cfg.ai.context_window}, "
            f"2*max_output_tokens={2 * cfg.ai.max_output_tokens}, "
            f"prompt_overhead={_PROMPT_OVERHEAD_TOKENS}); "
            f"raise max_output_tokens or context_window"
        )

    # Atomicity guard: a section alone > budget cannot be split. The
    # packing is greedy, so an oversized section is the only thing that
    # can produce a batch whose total exceeds the budget. Detect and fail
    # loud here, before the LLM call.
    for sec in sections:
        if sec.token_count > budget:
            raise PipelineError(
                f"Source {sec.source_id!r} section {sec.seq} alone "
                f"({sec.token_count} tokens) exceeds the artifact batch "
                f"budget ({budget} tokens); reduce section size or raise "
                f"context_window"
            )

    batches = _pack_sections(sections, budget)
    if len(batches) > _SOFT_COST_BATCH_THRESHOLD:
        logger.warning(
            "Artifact refine using %d batches (threshold=%d); consider "
            "raising context_window or selecting fewer sources",
            len(batches),
            _SOFT_COST_BATCH_THRESHOLD,
        )

    # ---- Single-batch path (byte-identical regression guard) ----
    if len(batches) == 1:
        transcript_text = _render_single_batch_text(sections)
        llm_prompt = _build_initial_prompt(
            prompt=prompt,
            transcript_text=transcript_text,
            cfg=cfg,
            ui_lang=ui_lang,
        )
        return await _call_llm_with_retry(llm_prompt, cfg, label="artifact batch 1/1")

    # ---- Multi-batch path ----
    return await _refine_artifact_multi_batch(
        prompt=prompt,
        batches=batches,
        cfg=cfg,
        ui_lang=ui_lang,
    )


async def _refine_artifact_multi_batch(
    *,
    prompt: str,
    batches: list[list[_SectionView]],
    cfg: BibilabConfig,
    ui_lang: str | None,
) -> ArtifactResult:
    """Multi-batch refine: batch 1 produces an initial draft; batch k>1
    feeds the running draft + new sections to the LLM with an
    'integrate' directive. The final ArtifactResult is the last call's
    parsed output.

    The new-sections text per batch is greedy-packed to ≤ budget tokens;
    the running draft is bounded by ``max_output_tokens`` (the model's
    hard ceiling on what it can produce, hence what can come back as a
    fed-back draft) and the prompt boilerplate by ``_PROMPT_OVERHEAD_TOKENS``.
    ContextWindowExceededError is re-raised by the retry ladder (it would
    re-overflow deterministically) and surfaces as a job-level failure.
    """
    draft: ArtifactResult | None = None
    for i, batch in enumerate(batches, start=1):
        label = f"artifact batch {i}/{len(batches)}"
        new_sections_text = "\n".join(_render_multi_batch_section(s) for s in batch)
        if i == 1:
            # First batch: same template as the single-batch path (single
            # source of truth via _build_initial_prompt), just with the
            # multi-batch section text. Any schema change in
            # _build_initial_prompt automatically flows here.
            llm_prompt = _build_initial_prompt(
                prompt=prompt,
                transcript_text=new_sections_text,
                cfg=cfg,
                ui_lang=ui_lang,
            )
        else:
            # Subsequent batches: refine the running draft.
            assert draft is not None  # invariant: set by i=1
            llm_prompt = _build_refine_prompt(
                prompt=prompt,
                draft=draft,
                new_sections_text=new_sections_text,
                cfg=cfg,
                ui_lang=ui_lang,
            )
        draft = await _call_llm_with_retry(llm_prompt, cfg, label=label)
    assert draft is not None  # invariant: at least one batch
    return draft


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

        try:
            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=10)

            # Cancellation check before LLM work.
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=30)

            # Section-batched refine: replaces the old single-call
            # _generate_artifact. _build_section_views handles source
            # existence + no-sections fail-loud.
            sections = await _build_section_views(source_ids)
            artifact_result = await _refine_artifact(
                prompt=prompt,
                sections=sections,
                cfg=cfg,
                ui_lang=meta_raw.get("ui_lang"),
            )

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
            content_path.write_text(artifact_result.content, encoding="utf-8")

            # Create artifact record with success status
            await create_artifact(
                artifact_id=artifact_id,
                list_id=list_id,
                name=artifact_result.name,
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

    async def _call_digest(
        self,
        transcript_text: str,
        video_meta: VideoMeta,
        cfg: BibilabConfig,
        ui_lang: str | None = None,
    ) -> DigestResult:
        """Run the digest LLM call. Does not catch exceptions — callers handle errors."""
        return await asyncio.to_thread(
            digest,
            transcript_text,
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

        transcript_text = await load_transcript_text(source_id)
        if not transcript_text:
            logger.warning("Digest job %s: source %s has no transcript", job_id, source_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error="Source has no transcript")
            return

        video_meta = VideoMeta.from_source(source)

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=10)

        try:
            extraction = await self._call_digest(transcript_text, video_meta, cfg, meta_raw.get("ui_lang"))
        except Exception as exc:
            logger.exception("Digest job %s failed", job_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
            return

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=80)

        await update_source_digest(
            source_id,
            extraction.summary,
            extraction.keywords,
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
            wav_path = await self._stage_extract_audio(job_id, video_path)
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
        extraction, sections = result

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
                detected_language,
                cfg,
                sentence_segments,
            )
        except Exception as exc:
            raise PipelineError(f"[persisting] {exc}") from exc
        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Job %s completed for video %s", job_id, video_id)

    async def _stage_download(
        self,
        job_id: str,
        video_meta: VideoMeta,
        source_id: str,
    ) -> Path | None:
        """Stage 1: Download video file and cover image."""
        await update_job_status(job_id, JobStatus.DOWNLOADING.value, progress=10)
        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, {"id": job_id})
            await delete_job(job_id)
            return None

        video_path: Path = await asyncio.to_thread(
            self._get_adapter().download,
            video_meta.video_id,
            video_meta.source_url,
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
    ) -> Path | None:
        """Stage 2: Extract audio from downloaded video."""
        await update_job_status(job_id, JobStatus.TRANSCRIBING.value, progress=25)
        try:
            wav_path = await asyncio.to_thread(extract_audio, video_path)
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
    ) -> tuple[DigestResult, list[Section]] | None:
        """Stage 4: Derive sections, chunk per-section, run digest + embed in parallel.

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

        meta_raw = parse_job_meta(job)
        transcript_text = format_turns(sentence_segments, include_time=False)

        async def _digest() -> DigestResult:
            return await self._call_digest(transcript_text, video_meta, cfg, meta_raw.get("ui_lang"))

        async def _embed() -> None:
            await asyncio.to_thread(embed_chunks, chunks, source_id, video_meta, list_id)

        extraction: DigestResult
        gather_results = await asyncio.gather(_digest(), _embed(), return_exceptions=True)
        extraction_raw, embed_raw = gather_results
        if isinstance(embed_raw, BaseException):
            logger.error("embed_chunks raised but was not the primary error", exc_info=embed_raw)
        if isinstance(extraction_raw, BaseException):
            raise extraction_raw
        if isinstance(embed_raw, BaseException):
            raise embed_raw
        extraction = extraction_raw

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await delete_job(job_id)
            return None

        return extraction, sections

    async def _stage_persist(
        self,
        job_id: str,
        source_id: str,
        video_id: str,
        video_meta: VideoMeta,
        list_id: str,
        extraction: DigestResult,
        sections: list[Section],
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
            source_id=source_id,
            video_id=video_id,
            platform=video_meta.platform,
            list_id=list_id,
            title=video_meta.title,
            summary=extraction.summary,
            keywords=extraction.keywords,
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
        for path in (self._bibilab_home / "downloads").glob(f"{video_id}.*"):
            path.unlink(missing_ok=True)
