"""THROWAWAY POC for the bounded-sections design (docs/specs/2026-06-07-bounded-sections-design.md).

NOT production code. Validates the foundation on a synthetic long transcript:
  1. chunk -> section derivation (token cap + coarse-pause snap, chunk-nested)
  2. refine summary chain (rolling context, per-section summary+keywords)
  3. boundedness (every section <= cap; refine input bounded) vs the single-digest baseline

Synthetic long transcript = real zh transcripts concatenated with monotonic
timestamps and a deliberate pause at each seam (a stand-in topic boundary).

Run:  cd backend && uv run python poc_bounded_sections.py
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from bibilab.config import bibilab_home, load_config
from bibilab.pipeline._shared import _call_llm, count_tokens
from bibilab.pipeline.chunk import RagChunk, chunk_segments
from bibilab.pipeline.transcribe import WhisperSegment

# --- POC knobs -------------------------------------------------------------
SECTION_CAP_TOKENS = 3000  # max tokens per section (real design: derive from context_window/max_output_tokens)
SECTION_MIN_RATIO = 0.5  # don't snap-cut a section below this fraction of cap
SECTION_PAUSE_S = 4.0  # coarse pause; bigger than chunk.py's 1.5s
SEAM_GAP_S = 5.0  # synthetic pause injected between concatenated sources
NUM_SOURCES = 6  # how many real sources to concatenate


@dataclass
class Section:
    seq: int
    seg_start: int
    seg_end: int
    text: str
    token_count: int
    ts_start: float
    ts_end: float
    n_chunks: int


# --- 1. Build a synthetic long transcript from real segments ---------------
def load_synthetic_transcript() -> tuple[list[WhisperSegment], list[int], list[str]]:
    """Concatenate NUM_SOURCES real transcripts; return (segments, seam_seg_indices, titles)."""
    db = sqlite3.connect(str(bibilab_home() / "bibilab.db"))
    db.row_factory = sqlite3.Row
    src_rows = db.execute(
        "SELECT s.id, s.title FROM sources s "
        "JOIN transcript_segments t ON t.source_id=s.id "
        "GROUP BY s.id ORDER BY COUNT(t.id) DESC LIMIT ?",
        (NUM_SOURCES,),
    ).fetchall()

    segs: list[WhisperSegment] = []
    seams: list[int] = []
    titles: list[str] = []
    offset = 0.0
    for sr in src_rows:
        titles.append(sr["title"])
        if segs:
            seams.append(len(segs))  # first seg index of a new source = a seam
        rows = db.execute(
            "SELECT start_s, end_s, text, speaker FROM transcript_segments WHERE source_id=? ORDER BY seq",
            (sr["id"],),
        ).fetchall()
        base = offset + (SEAM_GAP_S if segs else 0.0)
        for r in rows:
            segs.append(
                WhisperSegment(
                    start=base + r["start_s"],
                    end=base + r["end_s"],
                    text=r["text"],
                    speaker=r["speaker"],
                )
            )
        offset = segs[-1].end
    db.close()
    return segs, seams, titles


# --- 2. chunk -> section derivation (the new logic under test) -------------
def derive_sections(chunks: list[RagChunk]) -> list[Section]:
    """Greedily group consecutive chunks to SECTION_CAP_TOKENS, snapping the cut
    to the largest coarse-pause boundary inside the [min_ratio*cap, cap] window."""
    chunk_tokens = [count_tokens(c.text) for c in chunks]
    sections: list[Section] = []
    buf_idx: list[int] = []  # indices into chunks
    buf_tokens = 0

    def flush(idxs: list[int]) -> None:
        cs = [chunks[i] for i in idxs]
        sections.append(
            Section(
                seq=len(sections),
                seg_start=cs[0].seg_start,
                seg_end=cs[-1].seg_end,
                text=" ".join(c.text for c in cs),
                token_count=sum(chunk_tokens[i] for i in idxs),
                ts_start=cs[0].timestamp_start,
                ts_end=cs[-1].timestamp_end,
                n_chunks=len(cs),
            )
        )

    def gap_after(i: int, nxt: RagChunk | None) -> float:
        following = chunks[i + 1] if i + 1 < len(chunks) else nxt
        if following is None:
            return 0.0
        return following.timestamp_start - chunks[i].timestamp_end

    for ci, c in enumerate(chunks):
        ct = chunk_tokens[ci]
        if buf_idx and buf_tokens + ct > SECTION_CAP_TOKENS:
            # pick the split boundary within the window that maximizes pause gap
            cum = 0
            best_split = len(buf_idx) - 1  # default: cut right before the overflow chunk
            best_gap = gap_after(buf_idx[-1], c)
            for k, gi in enumerate(buf_idx):
                cum += chunk_tokens[gi]
                if cum < SECTION_MIN_RATIO * SECTION_CAP_TOKENS:
                    continue
                g = gap_after(gi, c)
                if g > best_gap:
                    best_gap, best_split = g, k
            flush(buf_idx[: best_split + 1])
            buf_idx = buf_idx[best_split + 1 :]
            buf_tokens = sum(chunk_tokens[i] for i in buf_idx)
        buf_idx.append(ci)
        buf_tokens += ct
    if buf_idx:
        flush(buf_idx)
    return sections


# --- 3. refine summary chain ----------------------------------------------
class SectionDigest(BaseModel):
    summary: str
    keywords: list[str]


_REFINE_PROMPT = """\
你在对一个长视频的转录逐段(section)做摘要,按顺序处理。读者已经看过前面各段的摘要,
不要重复前文已说的背景,只总结**当前段**的新内容。

前文各段摘要(上下文,勿复述):
{running}

当前段转录:
{section_text}

只输出一个 JSON 对象:
{{"summary": "60-100字,当前段的核心内容", "keywords": ["最多6个,每个1-4字词"]}}
用中文回答。"""


def refine_summaries(sections: list[Section], cfg) -> list[tuple[SectionDigest, int]]:
    """Sequential refine: each call sees prior summaries + current section.
    Returns [(digest, input_tokens)] so we can show input is bounded."""
    out: list[tuple[SectionDigest, int]] = []
    running_parts: list[str] = []
    for s in sections:
        running = "\n".join(running_parts) if running_parts else "(无,这是第一段)"
        prompt = _REFINE_PROMPT.format(running=running, section_text=s.text)
        in_tok = count_tokens(prompt)
        raw = _call_llm(prompt, cfg)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        dg = SectionDigest(**json.loads(raw))
        out.append((dg, in_tok))
        running_parts.append(f"[第{s.seq + 1}段] {dg.summary}")
    return out


_BASELINE_PROMPT = """\
为下面的长视频转录写一段 120-180 字的整体摘要,并给出最多 8 个关键词。
只输出 JSON: {{"summary": "...", "keywords": [...]}}。用中文回答。

转录:
{transcript}"""


def baseline_single_digest(full_text: str, cfg) -> tuple[SectionDigest, int]:
    prompt = _BASELINE_PROMPT.format(transcript=full_text)
    in_tok = count_tokens(prompt)
    raw = _call_llm(prompt, cfg)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return SectionDigest(**json.loads(raw)), in_tok


def main() -> None:
    cfg = load_config().ai
    segs, seams, titles = load_synthetic_transcript()
    full_text = " ".join(s.text for s in segs)
    total_tokens = count_tokens(full_text)

    print("=" * 78)
    print(f"SYNTHETIC TRANSCRIPT: {len(titles)} sources, {len(segs)} segments, ~{total_tokens} tokens")
    print(f"  seam segment indices (source boundaries): {seams}")
    for i, t in enumerate(titles):
        print(f"  src[{i}] {t[:46]}")

    chunks = chunk_segments(segs, language="zh")
    sections = derive_sections(chunks)

    print("=" * 78)
    print(f"CHUNKS: {len(chunks)}   ->   SECTIONS: {len(sections)}  (cap={SECTION_CAP_TOKENS} tok)")
    over = [s for s in sections if s.token_count > SECTION_CAP_TOKENS]
    print(f"  boundedness: max section = {max(s.token_count for s in sections)} tok; over-cap sections = {len(over)}")
    print(f"  section boundaries align w/ seams? section seg_starts: {[s.seg_start for s in sections]}")
    print(f"                                     seam seg indices : {seams}")
    print("-" * 78)
    for s in sections:
        print(
            f"  S{s.seq + 1}: segs[{s.seg_start:>3}-{s.seg_end:>3}] "
            f"{s.token_count:>4}tok  {s.n_chunks}chunks  "
            f"{int(s.ts_start) // 60}:{int(s.ts_start) % 60:02d}-{int(s.ts_end) // 60}:{int(s.ts_end) % 60:02d}"
        )

    print("=" * 78)
    print("REFINE CHAIN (per-section summary, rolling context):")
    refined = refine_summaries(sections, cfg)
    for s, (dg, in_tok) in zip(sections, refined):
        print("-" * 78)
        print(f"  S{s.seq + 1}  [refine input = {in_tok} tok]  kw={dg.keywords}")
        print(f"    {dg.summary}")

    print("=" * 78)
    print("BASELINE single digest over the WHOLE transcript (today's behavior):")
    base_dg, base_in = baseline_single_digest(full_text, cfg)
    print(f"  [input = {base_in} tok]  kw={base_dg.keywords}")
    print(f"  {base_dg.summary}")

    print("=" * 78)
    print("TOKEN ACCOUNTING:")
    refine_inputs = [it for _, it in refined]
    print(f"  refine: {len(refined)} calls, input tokens per call = {refine_inputs}")
    print(f"          max refine input = {max(refine_inputs)} tok  (BOUNDED, ~flat in source length)")
    print(f"  baseline: 1 call, input = {base_in} tok  (GROWS linearly with source length)")
    print(f"  union of section keywords = {sorted({k for _, (d, _) in zip(sections, refined) for k in d.keywords})}")
    print(f"  baseline keywords         = {sorted(base_dg.keywords)}")


if __name__ == "__main__":
    assert Path(bibilab_home() / "bibilab.db").exists(), "no bibilab.db"
    main()
