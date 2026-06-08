"""Digest generation - creates summary + keywords from transcript."""

import logging
import math

from pydantic import BaseModel, ValidationInfo, field_validator

from bibilab.adapters.base import VideoMeta
from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    _LANG_INSTRUCTION,
    _STRICT_SUFFIX,
    ContextWindowExceededError,
    LLMOutputBudgetExceededError,
    _call_llm,
    _lang_output_directive,
    _parse_llm_json_response,
    _resolved_lang,
)
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

# Keyword count: feeds the digest chip UI. The v2 chat prompt is static
# per language and no longer inlines a per-turn source list, so keywords
# are presentation-only. Kept short and topical; raised from 5 to widen
# query-topic coverage for the chip.
_MAX_KEYWORDS = 8


class SectionDigest(BaseModel):
    """Per-section digest: summary + keywords (no facets)."""

    summary: str
    keywords: list[str]


def _parse_section_digest_response(text: str) -> SectionDigest:
    """Parse an LLM JSON response into a SectionDigest, capping keywords."""
    sd: SectionDigest = _parse_llm_json_response(text, SectionDigest)
    sd.keywords = sd.keywords[:_MAX_KEYWORDS]
    return sd


def parse_facet_int(v: object) -> int | None:
    """Coerce a facet ordinal to an int >= 1.

    None / empty-string -> None ("unknown"). A present-but-unusable value
    (non-numeric, < 1, bool, non-finite float, wrong type) raises ValueError.
    The digest path catches that and degrades to None; the manual-edit path
    lets it surface as a 422 (a typed value is deliberate, not a guess).
    """
    if v is None:
        return None
    if isinstance(v, bool):  # bool is an int subclass; true/false is not an ordinal
        raise ValueError(f"not an integer: {v!r}")
    if isinstance(v, int):
        n = v
    elif isinstance(v, float) and math.isfinite(v) and v == int(v):
        n = int(v)
    elif isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            n = int(s)
        except ValueError as exc:
            raise ValueError(f"not an integer: {v!r}") from exc
    else:
        raise ValueError(f"not an integer: {v!r}")
    if n < 1:
        raise ValueError(f"must be >= 1: {n}")
    return n


def clean_str_facet(v: object) -> str | None:
    """Trim a string facet; '' / None -> None. Non-string raises ValueError.

    Used for series_name. The digest path catches the ValueError and degrades
    to None; the manual-edit path lets it 422.
    """
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"not a string: {v!r}")
    cleaned = v.strip()
    return cleaned or None


class DigestResult(BaseModel):
    summary: str
    keywords: list[str]
    series_name: str | None = None
    sequence_number: int | None = None
    season_number: int | None = None

    @field_validator("sequence_number", "season_number", mode="before")
    @classmethod
    def _coerce_int(cls, v: object, info: ValidationInfo) -> int | None:
        try:
            return parse_facet_int(v)
        except ValueError:
            logger.warning("digest: dropping unusable %s=%r", info.field_name, v)
            return None

    @field_validator("series_name", mode="before")
    @classmethod
    def _clean_str_facet(cls, v: object) -> str | None:
        try:
            return clean_str_facet(v)
        except ValueError:
            logger.warning("digest: dropping unusable series_name=%r", v)
            return None


_DIGEST_PROMPT = """\
Extract structured metadata from the video below. Return ONE JSON object and
nothing else. Each field is described once; follow its rule exactly.

summary (string)
  One paragraph, ~120-180 words, covering the main ideas.

keywords (list of strings, up to {max_keywords}, each 1-4 words)
  The main subjects the video covers, taken from title and transcript. Each
  keyword must be a self-contained topic a viewer could ask a question about
  on its own: a concrete dish, product, person, place, work, concept,
  theory, or technique.
  Exclude three kinds of non-topics:
    - format / meta words   e.g. 教程, 讲解, 合集, vlog, 视频
    - standalone modifiers  e.g. 麻辣, 韩式, 高清
    - labels too generic to identify this video   e.g. 美食, 知识
  Avoid near-duplicates (keep the most specific form only).
  Examples across genres (titles shown for brevity; the transcript adds more):
    "10分钟搞定！韩式炸鸡 + 辣鸡面"   -> 韩式炸鸡, 辣鸡面   (drop "10分钟搞定")
    "罗翔讲刑法：正当防卫的边界"      -> 罗翔, 刑法, 正当防卫   (drop "讲")
    "iPhone 16 Pro 深度评测 影像与续航" -> iPhone 16 Pro, 影像, 续航 (drop "深度评测")

series_name (string or null)
  The show / column identity, distinct from this video's title and the
  uploader name. e.g. "罗翔说刑法". null if none is stated.

sequence_number (integer or null)
  The video's ordinal within the series.
  e.g. "第8集"->8, "第12期"->12, "Chapter 5"->5, "上篇/中篇/下篇"->1/2/3.
  null if no ordinal is stated.

season_number (integer or null)
  e.g. "S2"->2, "第2季"->2. Usually null.

Rule for the three series fields: extract ONLY a value explicitly stated in
the title or transcript. Never guess. When unsure, use null - a wrong value
silently hides correct content later, while null just falls back to normal.

Output JSON in exactly this shape:
{{
  "summary": "string",
  "keywords": ["string", ...],
  "series_name": "string | null",
  "sequence_number": "integer | null",
  "season_number": "integer | null"
}}

Video title: {title}
Transcript:
{transcript}
"""


# Section index marker for the refine prompt's rolling context. Single
# language-agnostic format, consistent with the chat system's [N] citation
# convention. The marker is for the LLM's context disambiguation, not part
# of the output schema.
_REFINE_SECTION_MARKER = "[S{n}]"


_REFINE_PROMPT = """\
You are summarizing a long video's transcript section-by-section, in order.
The reader has seen prior section summaries; do not repeat prior background.
Summarize only the NEW content in the current section.

Prior section summaries (context, do not restate):
{running}

Current section transcript:
{section_text}

Output ONE JSON object:
{{"summary": "60-100 words covering this section's core content",
  "keywords": ["up to 8, each 1-4 words"]}}
"""


def _parse_response(text: str) -> DigestResult:
    result: DigestResult = _parse_llm_json_response(text, DigestResult)
    result.keywords = result.keywords[:_MAX_KEYWORDS]
    return result


def digest(
    transcript_text: str,
    meta: VideoMeta,
    cfg: AIConfig,
    output_language: str = "ui",
    ui_lang: str | None = None,
    llm_timeout: int = 120,
) -> DigestResult:
    lang = _resolved_lang(output_language, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])

    prompt = (
        lang_instruction
        + "\n\n"
        + _DIGEST_PROMPT.format(title=meta.title, transcript=transcript_text, max_keywords=_MAX_KEYWORDS)
        + f"\n\n{lang_instruction}\n{_lang_output_directive(lang)}"
    )

    # One plain attempt, then strict-suffix retries. Linear so the attempt
    # count is unambiguous. Only summary/keywords failures land here now —
    # facets degrade to None in the validators and never raise.
    prompts = [prompt, prompt + _STRICT_SUFFIX, prompt + _STRICT_SUFFIX]
    last_exc: Exception | None = None
    for attempt, p in enumerate(prompts, start=1):
        try:
            raw = _call_llm(p, cfg, llm_timeout=llm_timeout)
            return _parse_response(raw)
        except (ContextWindowExceededError, LLMOutputBudgetExceededError):
            # Structural failure — input overflow won't shrink on retry, and
            # output budget exhaustion is identical across attempts (same
            # prompt + same budget). Surface immediately rather than burning
            # two more calls. Transient errors fall through to the generic
            # except below and get retried.
            raise
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "LLM digest failed for %s (attempt %d/%d): %s",
                meta.video_id,
                attempt,
                len(prompts),
                exc,
            )

    # All retries exhausted — raise PipelineError instead of silent data loss
    error_msg = f"LLM digest exhausted all retries for {meta.video_id}: {last_exc}"
    logger.error(error_msg)
    raise PipelineError(error_msg)


def _build_refine_prompt(running: str, section_text: str, lang: str) -> str:
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    return (
        lang_instruction
        + "\n\n"
        + _REFINE_PROMPT.format(running=running, section_text=section_text)
        + f"\n\n{lang_instruction}\n{_lang_output_directive(lang)}"
    )


def _refine_one_section(
    section_text: str,
    running: str,
    lang: str,
    meta: VideoMeta,
    cfg: AIConfig,
    llm_timeout: int,
) -> SectionDigest:
    """Run the refine LLM call with the same retry ladder as digest()."""
    base_prompt = _build_refine_prompt(running=running, section_text=section_text, lang=lang)
    prompts = [base_prompt, base_prompt + _STRICT_SUFFIX, base_prompt + _STRICT_SUFFIX]
    last_exc: Exception | None = None
    for attempt, p in enumerate(prompts, start=1):
        try:
            raw = _call_llm(p, cfg, llm_timeout=llm_timeout)
            return _parse_section_digest_response(raw)
        except (ContextWindowExceededError, LLMOutputBudgetExceededError):
            raise
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "LLM section refine failed for %s (attempt %d/%d): %s",
                meta.video_id,
                attempt,
                len(prompts),
                exc,
            )

    error_msg = f"LLM section refine exhausted all retries for {meta.video_id}: {last_exc}"
    logger.error(error_msg)
    raise PipelineError(error_msg)


def digest_sections(
    section_texts: list[str],
    meta: VideoMeta,
    cfg: AIConfig,
    output_language: str = "ui",
    ui_lang: str | None = None,
    llm_timeout: int = 120,
) -> tuple[DigestResult, list[SectionDigest]]:
    """Run digest on each section: section 1 via digest() (with facets),
    sections 2..N via a lighter refine prompt with rolling context.

    `len(section_texts) == 0` is undefined behavior — derive_sections only
    returns [] for empty segments, which already failed upstream.

    Returns (DigestResult, list[SectionDigest]). For 1-section sources,
    the [SectionDigest] mirrors the DigestResult exactly (byte-identical
    contract).
    """
    lang = _resolved_lang(output_language, ui_lang)

    # Section 1: existing digest() — extracts facets once.
    extraction = digest(section_texts[0], meta, cfg, output_language, ui_lang, llm_timeout)
    section_digests: list[SectionDigest] = [SectionDigest(summary=extraction.summary, keywords=extraction.keywords)]

    # Sections 2..N: refine chain with rolling context.
    if len(section_texts) > 1:
        running_parts: list[str] = [_REFINE_SECTION_MARKER.format(n=1) + " " + extraction.summary]
        for i, text in enumerate(section_texts[1:], start=2):
            current_marker = _REFINE_SECTION_MARKER.format(n=i)
            running = "\n".join(running_parts + [current_marker])
            sd = _refine_one_section(
                section_text=text,
                running=running,
                lang=lang,
                meta=meta,
                cfg=cfg,
                llm_timeout=llm_timeout,
            )
            section_digests.append(sd)
            running_parts.append(_REFINE_SECTION_MARKER.format(n=i) + " " + sd.summary)

    return extraction, section_digests
