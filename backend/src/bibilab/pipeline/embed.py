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
from bibilab.config import BibilabConfig, bibilab_home, models_dir
from bibilab.db import _escape_fts_query, get_db_path, get_video_ids_for_sources, query_fts_rows
from bibilab.models._enums import CHAT_MODE_BROAD, CHAT_MODE_FOCUSED, ChatMode, RetrievalParams
from bibilab.pipeline.chunk import RagChunk

logger = logging.getLogger(__name__)


def _dynamic_pool(num_sources: int) -> int:
    """Candidate pool size that scales with source count.

    Floor of 10 (avoids under-sampling tiny lists), ceiling of 60 (bounds
    ChromaDB + FTS latency on huge lists), factor of 3× sources in between.
    """
    return min(max(num_sources * 3, 10), 60)


_chroma_collections: dict[str, "chromadb.Collection"] = {}


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
    # lower = more relevant (stores -score after RRF/rerank)
    best_score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    mode: ChatMode
    candidates_evaluated: int
    sources_with_hits: int
    sources_total: int
    source_coverage: list[SourceHit]


def _chunk_score(chunk: RetrievedChunk) -> float:
    """Canonical relevance where lower = more relevant."""
    return -chunk.score if chunk.score is not None else chunk.distance


def _chunk_key(chunk: RetrievedChunk) -> str:
    """Stable dedup key for a RetrievedChunk."""
    return f"{chunk.video_id}_{chunk.timestamp_start}_{chunk.timestamp_end}"


def _best_by_source(chunks: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    """Pick the most relevant chunk per video_id."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        vid = chunk.video_id
        if vid not in best or _chunk_score(chunk) < _chunk_score(best[vid]):
            best[vid] = chunk
    return best


def _diverse_top_k(ranked: list[RetrievedChunk], depth: int, k: int) -> list[RetrievedChunk]:
    """Pick up to k chunks, allowing at most depth chunks from any single video.

    Fills slots greedily in ranked order. If k slots can't be filled while
    respecting the depth cap (too few distinct sources), the cap is relaxed
    and leftovers fill the remaining slots.
    """
    per_src: dict[str, int] = {}
    picked: list[RetrievedChunk] = []
    leftovers: list[RetrievedChunk] = []

    for c in ranked:
        cur = per_src.get(c.video_id, 0)
        if cur < depth:
            picked.append(c)
            per_src[c.video_id] = cur + 1
            if len(picked) == k:
                return picked
        else:
            leftovers.append(c)

    return (picked + leftovers)[:k]


def _rrf_fuse(
    a: list[RetrievedChunk],
    b: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion on two ranked RetrievedChunk lists.

    RRF_score(d) = sum(1 / (60 + rank_i(d)) for each list where d appears).
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
        score = sum(1.0 / (60 + r) for r in ranks)
        chunk = key_to_chunk[key]
        chunk.score = score
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored]


def _embedding_model_dir() -> Path:
    return models_dir("embedding")


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
    global _chroma_collections
    name = cfg.transcript_collection_name
    if name not in _chroma_collections:
        import chromadb  # noqa: PLC0415

        client = chromadb.PersistentClient(path=str(bibilab_home() / "chroma"))
        _chroma_collections[name] = client.get_or_create_collection(
            name,
            embedding_function=_default_embedding_function(),
        )
    return _chroma_collections[name]


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
    chroma_path = bibilab_home() / "chroma"
    if not chroma_path.exists():
        return
    collection = _get_collection(cfg)
    try:
        collection.delete(where={"video_id": video_id})
    except Exception as exc:
        logger.warning("Failed to delete embeddings for video %s: %s", video_id, exc)


async def _resolve_video_ids(
    source_ids: list[str],
    video_ids: list[str] | None,
) -> list[str] | None:
    """Resolve video_ids from source_ids if not already provided."""
    if video_ids is not None:
        return video_ids
    if not source_ids:
        return None
    id_to_video_id = await get_video_ids_for_sources(source_ids)
    if not id_to_video_id:
        return None
    return list(id_to_video_id.values())


async def query_chunks(
    query_text: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    top_k: int = 10,
    *,
    video_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Query ChromaDB for transcript chunks relevant to the query, filtered by source.

    Args:
        query_text: The query string to search for relevant chunks.
        source_ids: List of source UUIDs to filter chunks by.
        cfg: BibilabConfig instance.
        top_k: Maximum number of chunks to return.
        video_ids: Optional pre-resolved video IDs. If not provided, resolved from source_ids.

    Returns:
        List of RetrievedChunk sorted by distance ascending (most relevant first).
        Returns empty list when no sources match, all results are below the
        relevance floor, or ChromaDB errors.
    """
    resolved = await _resolve_video_ids(source_ids, video_ids)
    if resolved is None:
        return []
    video_ids = resolved

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
    *,
    video_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Query FTS5 index for transcript chunks matching the query, filtered by source.

    Args:
        query_text: The query string to search for.
        source_ids: List of source UUIDs to filter by.
        cfg: BibilabConfig instance.
        top_k: Maximum number of chunks to return.
        video_ids: Optional pre-resolved video IDs. If not provided, resolved from source_ids.

    Returns list of RetrievedChunk with score set to the negated FTS5 BM25 rank
    (positive, higher = more relevant) and distance set to 0.0. _chunk_score()
    negates the score to recover "lower = more relevant" ordering.
    """
    resolved = await _resolve_video_ids(source_ids, video_ids)
    if resolved is None:
        return []
    video_ids = resolved

    escaped = _escape_fts_query(query_text)
    rows = await query_fts_rows(escaped, video_ids, top_k)

    return [
        RetrievedChunk(
            content=row["content"],
            video_title=row["video_title"],
            timestamp_start=float(row["timestamp_start"]),
            timestamp_end=float(row["timestamp_end"]),
            video_id=row["video_id"],
            distance=0.0,
            score=-row["rank"],
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
    video_ids = await _resolve_video_ids(source_ids, None)
    if video_ids is None:
        return []

    vec_task = query_chunks(query_text, source_ids, cfg, top_k=effective_top_k, video_ids=video_ids)
    fts_task = query_fts(query_text, source_ids, cfg, top_k=effective_top_k, video_ids=video_ids)

    vec_result, fts_result = await asyncio.gather(vec_task, fts_task, return_exceptions=True)

    if isinstance(vec_result, Exception):
        logger.warning("Vector query failed in hybrid search, falling back to FTS-only: %s", vec_result)
        vec_result = []

    if isinstance(fts_result, Exception):
        logger.warning("FTS query failed in hybrid search, falling back to vector-only: %s", fts_result)
        fts_result = []

    # RRF fusion only when both sides contributed; otherwise return the non-empty
    # ranking as-is to preserve raw distance semantics.
    if not vec_result:
        return fts_result
    if not fts_result:
        return vec_result
    return _rrf_fuse(vec_result, fts_result)


async def retrieve(
    query_text: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    params: RetrievalParams,
) -> RetrievalResult:
    """High-level retrieval that wraps hybrid search with metadata.

    Selection is driven by RetrievalParams: depth_per_source caps chunks from
    any single video; top_k sets the total returned.
    """
    sources_total = len(source_ids)
    effective_top_k = max(_dynamic_pool(sources_total), params.top_k)

    if cfg.rag.hybrid_enabled:
        chunks = await hybrid_search(query_text, source_ids, cfg, effective_top_k=effective_top_k)
    else:
        chunks = await query_chunks(query_text, source_ids, cfg, top_k=effective_top_k)

    candidates_evaluated = len(chunks)

    if cfg.rag.reranking_enabled and chunks:
        from bibilab.pipeline.rerank import rerank  # noqa: PLC0415

        try:
            chunks = await rerank(query_text, chunks, top_k=len(chunks))
        except Exception as exc:  # noqa: BLE001 - model load can fail in many ways
            logger.warning("Reranking failed, falling back to un-reranked chunks: %s", exc)

        pool_best_by_source = _best_by_source(chunks)

        floor = cfg.rag.rerank_min_score
        if floor is not None:
            chunks = [c for c in chunks if c.score is None or c.score >= floor]
    else:
        pool_best_by_source = _best_by_source(chunks)

    source_coverage = [
        SourceHit(
            video_id=c.video_id,
            video_title=c.video_title,
            best_score=_chunk_score(c),
        )
        for c in sorted(pool_best_by_source.values(), key=_chunk_score)
    ]

    result_chunks = _diverse_top_k(chunks, params.depth_per_source, params.top_k)

    inferred_mode: ChatMode = CHAT_MODE_BROAD if params.depth_per_source == 1 else CHAT_MODE_FOCUSED

    return RetrievalResult(
        chunks=result_chunks,
        mode=inferred_mode,
        candidates_evaluated=candidates_evaluated,
        sources_with_hits=len(pool_best_by_source),
        sources_total=sources_total,
        source_coverage=source_coverage,
    )
