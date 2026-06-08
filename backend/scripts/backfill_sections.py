"""One-shot backfill: derive `sections` rows for every existing source.

Run from `backend/`:
    uv run python -m scripts.backfill_sections            # commit
    uv run python -m scripts.backfill_sections --dry-run  # print plan only

Idempotent: each source's existing section rows are replaced (DELETE+INSERT).
No re-embed: all current sources are short → 1 section per source → existing
chunks already nest in it (the algorithm guarantees containment by
construction, and chunks are produced per-section since #452).

Exit code 0 on success. Designed to be safe to re-run.
"""

import argparse
import asyncio
import logging
import sqlite3
from pathlib import Path

from bibilab.db import get_db_path, write_sections
from bibilab.pipeline.section import derive_sections
from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)


def _load_segments_sync(db_path: Path) -> list[tuple[str, list[WhisperSegment]]]:
    """Sync read of (source_id, [segments]) for all sources. One DB hit, no
    async machinery — backfill is a one-shot CLI tool, not a hot path."""
    out: list[tuple[str, list[WhisperSegment]]] = []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Order by source_id so the output is deterministic across runs.
        source_ids = [r["id"] for r in conn.execute("SELECT id FROM sources ORDER BY id").fetchall()]
        for sid in source_ids:
            rows = conn.execute(
                "SELECT start_s, end_s, text, speaker FROM transcript_segments WHERE source_id = ? ORDER BY seq",
                (sid,),
            ).fetchall()
            segs = [
                WhisperSegment(start=r["start_s"], end=r["end_s"], text=r["text"], speaker=r["speaker"]) for r in rows
            ]
            out.append((sid, segs))
    finally:
        conn.close()
    return out


async def run(dry_run: bool = False) -> int:
    db_path = get_db_path()
    if not db_path.exists():
        print(f"No database at {db_path} — nothing to backfill.")
        return 0

    sources = _load_segments_sync(db_path)
    print(f"Found {len(sources)} source(s) to process ({'DRY RUN — no writes' if dry_run else 'will write sections'}).")

    total_sections = 0
    for sid, segs in sources:
        secs = derive_sections(segs, language="en")
        total_sections += len(secs)
        print(f"  {sid}: {len(segs)} segments → {len(secs)} section(s)")
        if not dry_run:
            await write_sections(sid, secs)

    print(f"Done. {len(sources)} source(s) → {total_sections} section(s) total.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing section rows.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(asyncio.run(run(dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
