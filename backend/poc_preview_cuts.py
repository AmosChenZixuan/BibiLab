"""THROWAWAY — preview cut positions for TOKEN~6000 + PAUSE. No model/LLM.

Prints each section's span + the actual transcript text on both sides of every
cut, plus the pause gap and distance to the nearest synthetic seam, so the cut
quality is eyeballable.

Run:  cd backend && uv run python poc_preview_cuts.py
"""

import sqlite3

from bibilab.config import bibilab_home
from bibilab.pipeline._shared import count_tokens
from bibilab.pipeline.transcribe import WhisperSegment

NUM_SOURCES = 6
TARGET_TOKEN = 6000
ZONE_LOW, ZONE_HIGH = 0.6, 1.4


def load():
    db = sqlite3.connect(str(bibilab_home() / "bibilab.db"))
    db.row_factory = sqlite3.Row
    rows_src = db.execute(
        "SELECT s.id, s.title FROM sources s JOIN transcript_segments t ON t.source_id=s.id "
        "GROUP BY s.id ORDER BY COUNT(t.id) DESC LIMIT ?",
        (NUM_SOURCES,),
    ).fetchall()
    segs, seams, titles = [], [], []
    offset = 0.0
    for sr in rows_src:
        titles.append(sr["title"])
        if segs:
            seams.append(len(segs))
        rows = db.execute(
            "SELECT start_s, end_s, text, speaker FROM transcript_segments WHERE source_id=? ORDER BY seq",
            (sr["id"],),
        ).fetchall()
        base = offset
        for r in rows:
            segs.append(
                WhisperSegment(start=base + r["start_s"], end=base + r["end_s"], text=r["text"], speaker=r["speaker"])
            )
        offset = segs[-1].end
    db.close()
    return segs, seams, titles


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main():
    segs, seams, titles = load()
    seg_tok = [count_tokens(s.text) for s in segs]
    n = len(segs)
    pause = [segs[i + 1].start - segs[i].end for i in range(n - 1)] + [0.0]

    # sections: token target + max-pause snap inside zone
    secs, cuts, start = [], [], 0
    while start < n:
        best_i, best_p, forced = None, -1.0, n - 1
        for i in range(start, n):
            tok = sum(seg_tok[start : i + 1])
            if tok < ZONE_LOW * TARGET_TOKEN:
                continue
            if tok > ZONE_HIGH * TARGET_TOKEN:
                forced = i
                break
            if i < n - 1 and pause[i] > best_p:
                best_p, best_i = pause[i], i
        cut = best_i if best_i is not None else min(forced, n - 1)
        secs.append((start, cut))
        if cut < n - 1:
            cuts.append(cut)
        if cut >= n - 1:
            break
        start = cut + 1

    print("=" * 90)
    print(f"{len(titles)} eps, {n} segs, {mmss(segs[-1].end)} long. Seams (true topic edges): {seams}")
    print(f"TOKEN~6000 + PAUSE  ->  {len(secs)} sections, {len(cuts)} cuts: {cuts}")
    print("=" * 90)
    for k, (a, b) in enumerate(secs):
        tok = sum(seg_tok[a : b + 1])
        print(f"\nS{k + 1}  segs[{a}-{b}]  {tok} tok  {mmss(segs[a].start)}-{mmss(segs[b].end)}")
        if b < n - 1:
            i = b  # cut is after seg i=b
            d = min(abs(i - s) for s in seams)
            nearest = min(seams, key=lambda s: abs(i - s))
            print(
                f"   ──CUT after seg {i}── pause={pause[i]:.1f}s  @{mmss(segs[i].end)}"
                f"  | nearest seam {nearest} (dist {d} segs)"
            )
            for j in (b - 1, b):
                if 0 <= j < n:
                    print(f"      end {j:>3}: {segs[j].text[:48]}")
            for j in (b + 1, b + 2):
                if 0 <= j < n:
                    mark = "  << next episode starts here" if (b + 1) in seams and j == b + 1 else ""
                    print(f"      nxt {j:>3}: {segs[j].text[:48]}{mark}")


if __name__ == "__main__":
    main()
