"""One-shot: re-tokenize chunks_fts to bigram tokens. Idempotent (per-row).

Run once after deploying the bigram tokenization change. Safe to re-run —
already-migrated rows are detected and skipped.
"""

import sqlite3

from bibilab.db import _CJK, _tokenize_cjk, get_db_path


def _is_migrated(content: str) -> bool:
    """Old format has one CJK char per token; new format has 2-CJK-char bigrams."""
    return any(len(t) == 2 and _CJK.match(t[0]) and _CJK.match(t[1]) for t in content.split())


def main() -> None:
    db_path = str(get_db_path())
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT rowid, content FROM chunks_fts").fetchall()
    migrated = 0
    skipped = 0
    for rowid, content in rows:
        if _is_migrated(content):
            skipped += 1
            continue
        conn.execute(
            "UPDATE chunks_fts SET content = ? WHERE rowid = ?",
            (_tokenize_cjk(content), rowid),
        )
        migrated += 1
    conn.commit()
    conn.close()
    print(f"Migrated {migrated} rows, skipped {skipped} (already migrated)")


if __name__ == "__main__":
    main()
