"""THROWAWAY POC #2 — section boundary strategies (docs/specs/2026-06-07-bounded-sections-design.md).

Compares where to cut a long transcript into sections. Cuts ALWAYS land on a
segment boundary (never mid-sentence). Two target intervals x two snap signals:

  target:  TIME (~30 min)   |  TOKEN (~6000 tok)
  signal:  longest PAUSE    |  max topic-DRIFT (embedding cosine)

Synthetic long transcript = real zh transcripts concatenated with CONTINUOUS
timestamps (no injected pause), so the episode seams are genuine topic shifts
detectable by drift but NOT specially marked by pause (the realistic case).
The seams are the ground truth for "did this strategy find the topic boundary".

Run:  cd backend && uv run python poc_section_boundaries.py
"""

import sqlite3
from dataclasses import dataclass

import numpy as np

from bibilab.config import bibilab_home
from bibilab.pipeline._shared import count_tokens
from bibilab.pipeline.embed import ONNXMultilingualEmbedding
from bibilab.pipeline.transcribe import WhisperSegment

NUM_SOURCES = 6
TARGET_TIME_S = 30 * 60
TARGET_TOKEN = 6000
ZONE_LOW, ZONE_HIGH = 0.6, 1.4  # cut-zone around target
DRIFT_WINDOW = 8  # segments each side for topic-drift cosine


def load_synthetic_transcript() -> tuple[list[WhisperSegment], list[int], list[str]]:
    db = sqlite3.connect(str(bibilab_home() / "bibilab.db"))
    db.row_factory = sqlite3.Row
    src_rows = db.execute(
        "SELECT s.id, s.title FROM sources s JOIN transcript_segments t ON t.source_id=s.id "
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
            seams.append(len(segs))
        rows = db.execute(
            "SELECT start_s, end_s, text, speaker FROM transcript_segments WHERE source_id=? ORDER BY seq",
            (sr["id"],),
        ).fetchall()
        base = offset  # CONTINUOUS: no injected seam gap
        for r in rows:
            segs.append(
                WhisperSegment(start=base + r["start_s"], end=base + r["end_s"], text=r["text"], speaker=r["speaker"])
            )
        offset = segs[-1].end
    db.close()
    return segs, seams, titles


def compute_signals(segs: list[WhisperSegment]) -> tuple[np.ndarray, np.ndarray]:
    """Return (pause_gap[i], drift[i]) for boundary AFTER segment i (i in 0..n-2)."""
    n = len(segs)
    pause = np.array([segs[i + 1].start - segs[i].end for i in range(n - 1)])

    emb_fn = ONNXMultilingualEmbedding()
    vecs: list[list[float]] = []
    for s in range(0, n, 64):
        vecs.extend(emb_fn([seg.text for seg in segs[s : s + 64]]))
    E = np.array(vecs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    drift = np.zeros(n - 1)
    for i in range(n - 1):
        left = E[max(0, i - DRIFT_WINDOW + 1) : i + 1].mean(axis=0)
        right = E[i + 1 : min(n, i + 1 + DRIFT_WINDOW)].mean(axis=0)
        ln, rn = left / (np.linalg.norm(left) + 1e-9), right / (np.linalg.norm(right) + 1e-9)
        drift[i] = 1.0 - float(ln @ rn)
    return pause, drift


@dataclass
class Sec:
    seg_start: int
    seg_end: int
    dur_s: float
    tokens: int


def make_sections(segs, seg_tok, signal_arr, measure: str, target: float) -> list[Sec]:
    """Greedy: accumulate `measure` from section start; inside the [low,high]*target
    zone, cut at the segment boundary maximizing `signal_arr`. Cut is on a segment."""
    n = len(segs)
    out: list[Sec] = []
    start = 0
    while start < n:
        best_i, best_sig = None, -1e9
        forced_i = n - 1
        for i in range(start, n):
            m = (segs[i].end - segs[start].start) if measure == "time" else sum(seg_tok[start : i + 1])
            if m < ZONE_LOW * target:
                continue
            if m > ZONE_HIGH * target:
                forced_i = i
                break
            if i < n - 1 and signal_arr[i] > best_sig:  # boundary AFTER seg i
                best_sig, best_i = signal_arr[i], i
        cut = best_i if best_i is not None else min(forced_i, n - 1)
        out.append(Sec(start, cut, segs[cut].end - segs[start].start, sum(seg_tok[start : cut + 1])))
        if cut >= n - 1:
            break
        start = cut + 1
    return out


def seam_alignment(cuts: list[int], seams: list[int]) -> tuple[float, int]:
    """Mean |cut - nearest seam| (segments) + #seams covered within 5 segs of a cut."""
    if not cuts:
        return float("nan"), 0
    mean_d = float(np.mean([min(abs(c - s) for s in seams) for c in cuts]))
    covered = sum(1 for s in seams if any(abs(c - s) <= 5 for c in cuts))
    return mean_d, covered


def main() -> None:
    segs, seams, titles = load_synthetic_transcript()
    seg_tok = [count_tokens(s.text) for s in segs]
    total_tok = sum(seg_tok)
    total_min = segs[-1].end / 60
    print("=" * 84)
    print(
        f"SYNTHETIC: {len(titles)} eps, {len(segs)} segs, ~{total_tok} tok,"
        f" {total_min:.0f} min (continuous, no injected pause)"
    )
    print(f"  seam seg indices (true topic boundaries): {seams}")

    pause, drift = compute_signals(segs)
    top_pause = sorted(range(len(pause)), key=lambda i: -pause[i])[:8]
    top_drift = sorted(range(len(drift)), key=lambda i: -drift[i])[:8]
    print("-" * 84)
    print(f"  top-8 PAUSE boundaries (seg idx): {sorted(top_pause)}  max={pause.max():.1f}s")
    print(f"  top-8 DRIFT boundaries (seg idx): {sorted(top_drift)}")
    print(f"  -> do top signals land near seams {seams}?  (drift should; pause shouldn't, no injected gaps)")

    configs = [
        ("TIME~30min  + PAUSE", pause, "time", TARGET_TIME_S),
        ("TIME~30min  + DRIFT", drift, "time", TARGET_TIME_S),
        ("TOKEN~6000  + PAUSE", pause, "token", TARGET_TOKEN),
        ("TOKEN~6000  + DRIFT", drift, "token", TARGET_TOKEN),
    ]
    for name, sig, measure, target in configs:
        secs = make_sections(segs, seg_tok, sig, measure, target)
        cuts = [s.seg_end for s in secs[:-1]]  # internal boundaries
        mean_d, covered = seam_alignment(cuts, seams)
        print("=" * 84)
        print(f"{name}:  {len(secs)} sections")
        print(f"  durations(min): {[round(s.dur_s / 60, 1) for s in secs]}")
        print(f"  tokens        : {[s.tokens for s in secs]}")
        print(f"  cut seg idxs  : {cuts}")
        print(
            f"  seam alignment: mean cut->nearest-seam = {mean_d:.1f} segs; seams covered(<=5) = {covered}/{len(seams)}"
        )


if __name__ == "__main__":
    main()
