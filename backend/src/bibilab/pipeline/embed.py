"""ChromaDB chunk embedding step."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

import sqlite3
from pathlib import Path

from bibilab.adapters.base import VideoMeta
from bibilab.config import BibilabConfig, bibilab_home
from bibilab.db import get_db_path, get_video_ids_for_sources, query_fts_rows
from bibilab.pipeline.chunk import RagChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    content: str
    video_title: str
    timestamp_start: float
    timestamp_end: float
    video_id: str
    distance: float


def _embedding_model_dir() -> Path:
    return bibilab_home() / "models" / "embedding"


def is_embedding_model_downloaded() -> bool:
    """Return True if the ONNX embedding model files are present locally."""
    return (_embedding_model_dir() / "onnx" / "model.onnx").exists()


def _default_embedding_function() -> "ONNXMiniLM_L6_V2":
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # noqa: PLC0415

    class LocalONNXMiniLM(ONNXMiniLM_L6_V2):
        DOWNLOAD_PATH = _embedding_model_dir()

    _embedding_model_dir().mkdir(parents=True, exist_ok=True)
    return LocalONNXMiniLM()


def _get_collection(cfg: BibilabConfig) -> "chromadb.Collection":
    import chromadb  # noqa: PLC0415

    client = chromadb.PersistentClient(path=str(bibilab_home() / "chroma"))
    return client.get_or_create_collection(
        cfg.transcript_collection_name,
        embedding_function=_default_embedding_function(),
    )


def embed_chunks(
    chunks: list[RagChunk],
    meta: VideoMeta,
    list_id: str,
    cfg: BibilabConfig,
) -> None:
    if not chunks:
        return

    collection = _get_collection(cfg)

    # Remove any existing chunks for this video (idempotent re-run)
    try:
        collection.delete(where={"video_id": meta.video_id})
    except Exception as exc:
        logger.warning("Failed to delete existing embeddings for video %s: %s", meta.video_id, exc)

    ids = [f"{meta.video_id}_{chunk.sequence_index}" for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "video_id": meta.video_id,
            "list_id": list_id,
            "video_title": meta.title,
            "timestamp_start": chunk.timestamp_start,
            "timestamp_end": chunk.timestamp_end,
            "sequence_index": chunk.sequence_index,
        }
        for chunk in chunks
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Embedded %d chunks for %s", len(chunks), meta.video_id)

    populate_fts(chunks, meta)


def populate_fts(chunks: list[RagChunk], meta: VideoMeta) -> None:
    """Insert transcript chunks into the FTS5 index (sync, called from embed thread)."""
    if not chunks:
        return
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    try:
        # Clear existing FTS rows for this video (idempotent re-run)
        conn.execute("DELETE FROM chunks_fts WHERE video_id = ?", (meta.video_id,))
        conn.executemany(
            "INSERT INTO chunks_fts (content, video_id, video_title, timestamp_start, timestamp_end, chunk_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    chunk.text,
                    meta.video_id,
                    meta.title,
                    chunk.timestamp_start,
                    chunk.timestamp_end,
                    f"{meta.video_id}_{chunk.sequence_index}",
                )
                for chunk in chunks
            ],
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("FTS indexed %d chunks for %s", len(chunks), meta.video_id)


def clear_fts_for_video_sync(video_id: str) -> None:
    """Delete all FTS rows for a video (sync, for use from worker threads)."""
    db_path = get_db_path()
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM chunks_fts WHERE video_id = ?", (video_id,))
        conn.commit()
    finally:
        conn.close()


def clear_embeddings_for_list(list_id: str, cfg: BibilabConfig) -> None:
    """Delete all ChromaDB chunks belonging to the given list."""
    collection = _get_collection(cfg)
    try:
        collection.delete(where={"list_id": list_id})
    except Exception as exc:
        logger.warning("Failed to delete embeddings for list %s: %s", list_id, exc)


def clear_embeddings_for_video(video_id: str, cfg: BibilabConfig) -> None:
    """Delete all ChromaDB chunks belonging to the given video."""
    collection = _get_collection(cfg)
    try:
        collection.delete(where={"video_id": video_id})
    except Exception as exc:
        logger.warning("Failed to delete embeddings for video %s: %s", video_id, exc)


async def query_chunks(
    query_text: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Query ChromaDB for transcript chunks relevant to the query, filtered by source.

    Args:
        query_text: The query string to search for relevant chunks.
        source_ids: List of source UUIDs to filter chunks by.
        cfg: BibilabConfig instance.
        top_k: Maximum number of chunks to return.

    Returns:
        List of RetrievedChunk sorted by distance ascending (most relevant first).
        Returns empty list when no sources match, all results are below the
        relevance floor, or ChromaDB errors.
    """
    if not source_ids:
        return []

    id_to_video_id = await get_video_ids_for_sources(source_ids)
    if not id_to_video_id:
        return []

    video_ids = list(id_to_video_id.values())

    def _sync_query() -> dict:
        collection = _get_collection(cfg)
        return collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where={"video_id": {"$in": video_ids}},
        )

    try:
        results = await asyncio.to_thread(_sync_query)
    except Exception as exc:  # noqa: BLE001 - ChromaDB errors vary by version
        logger.warning("ChromaDB query failed: %s", exc)
        return []

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not documents[0]:
        return []

    floor = cfg.rag.max_distance

    # ChromaDB returns results sorted by distance ascending; preserve order
    return [
        RetrievedChunk(
            content=documents[0][i],
            video_title=metadatas[0][i].get("video_title", ""),
            timestamp_start=metadatas[0][i].get("timestamp_start", 0.0),
            timestamp_end=metadatas[0][i].get("timestamp_end", 0.0),
            video_id=metadatas[0][i].get("video_id", ""),
            distance=distances[0][i],
        )
        for i in range(len(documents[0]))
        if distances[0][i] <= floor
    ]


async def query_fts(
    query_text: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    top_k: int = 30,
) -> list[RetrievedChunk]:
    """Query FTS5 index for transcript chunks matching the query, filtered by source.

    Returns list of RetrievedChunk with distance derived from BM25 rank.
    FTS5 rank is negative (more negative = better match); we negate it to get
    a positive distance-like score where lower = more relevant.
    """
    if not source_ids:
        return []

    id_to_video_id = await get_video_ids_for_sources(source_ids)
    if not id_to_video_id:
        return []

    video_ids = list(id_to_video_id.values())

    try:
        rows = await query_fts_rows(query_text, video_ids, top_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FTS query failed: %s", exc)
        return []

    return [
        RetrievedChunk(
            content=row["content"],
            video_title=row["video_title"],
            timestamp_start=float(row["timestamp_start"]),
            timestamp_end=float(row["timestamp_end"]),
            video_id=row["video_id"],
            distance=-row["rank"],  # negate: FTS5 rank is negative, lower = better
        )
        for row in rows
    ]
