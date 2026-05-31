#!/usr/bin/env python3
"""One-shot migration: add pinyin column to chunks_fts.

Reads Chroma documents (source of truth for raw chunk text), recomputes
content + pinyin FTS tokens, drops and recreates chunks_fts with the new
pinyin column, then rewrites all rows.

Usage: uv run python scripts/reindex_fts_pinyin.py

Safe: Chroma is read-only. FTS is rebuilt from Chroma data. If anything
goes wrong, re-ingest affected sources to repopulate FTS from scratch.
"""

from __future__ import annotations

import sqlite3
import sys

from bibilab.config import load_config
from bibilab.db import _pinyin_index_tokens, _tokenize_cjk, get_db_path
from bibilab.pipeline.embed import _get_collection

cfg = load_config()
db_path = get_db_path()
collection = _get_collection(cfg)

# Count current rows (table may not exist on fresh DB)
conn = sqlite3.connect(str(db_path))
try:
    before = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
except sqlite3.OperationalError:
    before = 0
conn.close()
print(f"Current FTS rows: {before}")

# Drop and recreate with new schema
conn = sqlite3.connect(str(db_path))
conn.execute("DROP TABLE IF EXISTS chunks_fts")
conn.execute(
    "CREATE VIRTUAL TABLE chunks_fts USING fts5("
    "content, pinyin,"
    "source_id UNINDEXED, video_title UNINDEXED,"
    "timestamp_start UNINDEXED, timestamp_end UNINDEXED,"
    "chunk_id UNINDEXED, seg_start UNINDEXED, seg_end UNINDEXED)"
)
conn.commit()
conn.close()
print("Recreated chunks_fts with pinyin column.")

# Reindex from ChromaDB
print("Fetching ChromaDB documents...")
results = collection.get(include=["documents", "metadatas"])
if not results["ids"]:
    print("No documents in ChromaDB. FTS is empty. Done.")
    sys.exit(0)

print(f"Found {len(results['ids'])} Chroma documents. Reindexing...")

conn = sqlite3.connect(str(db_path))
rows = []
skipped = 0
for i, doc_id in enumerate(results["ids"]):
    text = results["documents"][i]
    meta = results["metadatas"][i]
    if text is None or meta is None:
        skipped += 1
        continue
    rows.append(
        (
            _tokenize_cjk(str(text)),
            _pinyin_index_tokens(str(text)),
            meta.get("source_id", ""),
            meta.get("video_title", ""),
            float(meta.get("timestamp_start", 0.0)),
            float(meta.get("timestamp_end", 0.0)),
            doc_id,
            meta.get("seg_start"),
            meta.get("seg_end"),
        )
    )

conn.executemany(
    "INSERT INTO chunks_fts "
    "(content, pinyin, source_id, video_title, timestamp_start, timestamp_end, chunk_id, seg_start, seg_end) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    rows,
)
conn.commit()
after = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
conn.close()

print(f"Reindexed {after} rows ({skipped} skipped — null documents).")
print(f"Before: {before}, After: {after}")
print("Done.")
