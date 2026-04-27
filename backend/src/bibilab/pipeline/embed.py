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
    score: float | None = None


@dataclass
class SourceHit:
    video_id: str
    video_title: str
    best_distance: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    mode: str
    candidates_evaluated: int
    sources_with_hits: int
    sources_total: int
    source_coverage: list[SourceHit]


def _chunk_key(chunk: RetrievedChunk) -> str:
    """Stable dedup key for a RetrievedChunk."""
    return f"{chunk.video_id}_{chunk.timestamp_start}_{chunk.timestamp_end}"


def _rrf_fuse(
    a: list[RetrievedChunk],
    b: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion on two ranked RetrievedChunk lists.

    RRF_score(d) = sum(1 / (k + rank_i(d)) for each list where d appears).
    Returns fused list sorted by RRF score descending. Duplicates (same _chunk_key)
    are merged into a single entry.
    """
    positions: dict[str, list[int]] = {}
    key_to_chunk: dict[str, RetrievedChunk] = {}

    for source in (a, b):
        for rank, chunk in enumerate(source, start=1):
            key = _chunk_key(chunk)
            positions.setdefault(key, []).append(rank)
            key_to_chunk.setdefault(key, chunk)

    scored = []
    for key, ranks in positions.items():
        score = sum(1.0 / (k + r) for r in ranks)
        chunk = key_to_chunk[key]
        chunk.score = score
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored]


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


async def hybrid_search(
    query_text: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    effective_top_k: int,
) -> list[RetrievedChunk]:
    """Run vector and FTS5 BM25 retrieval in parallel, merge via RRF.

    Falls back to vector-only if FTS returns empty or errors.
    """
    vec_task = query_chunks(query_text, source_ids, cfg, top_k=effective_top_k)
    fts_task = query_fts(query_text, source_ids, cfg, top_k=effective_top_k)

    results = await asyncio.gather(vec_task, fts_task, return_exceptions=True)
    vec_chunks, fts_result = results[0], results[1]

    if isinstance(fts_result, Exception):
        logger.warning("FTS query failed in hybrid search, falling back to vector-only: %s", fts_result)
        return vec_chunks

    fts_chunks = fts_result

    if not fts_chunks:
        return vec_chunks

    return _rrf_fuse(vec_chunks, fts_chunks, k=60)


async def retrieve(
    query_text: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    mode: str = "focused",
    top_k: int = 10,
) -> RetrievalResult:
    """High-level retrieval that wraps hybrid search with metadata."""
    sources_total = len(source_ids)
    effective_top_k = 50 if mode == "broad" else top_k

    if cfg.rag.hybrid_enabled:
        chunks = await hybrid_search(query_text, source_ids, cfg, effective_top_k=effective_top_k)
    else:
        chunks = await query_chunks(query_text, source_ids, cfg, top_k=effective_top_k)

    if cfg.rag.reranking_enabled and chunks:
        from bibilab.pipeline.rerank import rerank  # noqa: PLC0415

        chunks = await rerank(query_text, chunks, top_k=top_k)

    def _score(chunk: RetrievedChunk) -> float:
        """Canonical relevance where lower = more relevant."""
        return -chunk.score if chunk.score is not None else chunk.distance

    best_by_source: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        vid = chunk.video_id
        if vid not in best_by_source or _score(chunk) < _score(best_by_source[vid]):
            best_by_source[vid] = chunk

    source_coverage = [
        SourceHit(
            video_id=c.video_id,
            video_title=c.video_title,
            best_distance=_score(c),
        )
        for c in sorted(best_by_source.values(), key=_score)
    ]

    result_chunks = sorted(best_by_source.values(), key=_score) if mode == "broad" else chunks

    return RetrievalResult(
        chunks=result_chunks,
        mode=mode,
        candidates_evaluated=len(chunks),
        sources_with_hits=len(best_by_source),
        sources_total=sources_total,
        source_coverage=source_coverage,
    )
