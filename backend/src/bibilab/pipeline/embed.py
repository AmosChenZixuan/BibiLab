"""ChromaDB chunk embedding step."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb

import sqlite3
from pathlib import Path

from bibilab.adapters.base import VideoMeta
from bibilab.config import BibilabConfig, bibilab_home, models_dir
from bibilab.db import _pinyin_index_tokens, _tokenize_cjk, get_db_path, query_fts_rows
from bibilab.model_registry import EMBEDDING_SPEC_ID, ensure
from bibilab.pipeline.chat_inference_pool import get_chat_pool
from bibilab.pipeline.chunk import RagChunk

logger = logging.getLogger(__name__)


_chroma_collections: dict[str, "chromadb.Collection"] = {}
# Serializes the lazy ChromaDB client construction. chromadb 1.x corrupts its
# global client state (RustBindingsAPI / "tenant default_tenant" errors) if two
# threads build the PersistentClient concurrently on a cold cache — which
# hybrid_search does via its parallel vector + FTS-backfill Chroma calls.
_chroma_lock = threading.Lock()


@dataclass
class RetrievedChunk:
    content: str
    video_title: str
    timestamp_start: float
    timestamp_end: float
    source_id: str
    distance: float
    score: float | None = None
    # None when chroma_id lacks the {source_id}_{N} suffix (malformed / legacy).
    sequence_index: int | None = None
    # Segment seq-range covered by this chunk (None for legacy/malformed rows).
    # Used by chat top-k reconstruction to fetch the speaker-turn body.
    seg_start: int | None = None
    seg_end: int | None = None


@dataclass
class SourceHit:
    source_id: str
    video_title: str
    # lower = more relevant (stores -score after RRF/rerank)
    best_score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    candidates_evaluated: int
    sources_with_hits: int
    sources_total: int
    source_coverage: list[SourceHit]
    # True when rerank ran successfully; useful for offline quality analysis
    # (rerank score != distance, so callers can disambiguate the two regimes).
    reranked: bool = False


def _chunk_score(chunk: RetrievedChunk) -> float:
    """Canonical relevance where lower = more relevant."""
    return -chunk.score if chunk.score is not None else chunk.distance


def _chunk_key(chunk: RetrievedChunk) -> str:
    """Stable dedup key for a RetrievedChunk."""
    return f"{chunk.source_id}_{chunk.timestamp_start}_{chunk.timestamp_end}"


def _parse_seq_index(chroma_id: str) -> int | None:
    """Parse sequence_index from a ChromaDB id of the form '{source_id}_{sequence_index}'."""
    try:
        return int(chroma_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _row_from_chroma(
    *,
    content: str,
    metadata: dict,
    distance: float,
    chroma_id: str,
    score: float | None = None,
) -> RetrievedChunk:
    """Build a RetrievedChunk from one Chroma row (id + document + metadata + distance).

    Shared by query_chunks (vector) and query_fts so row parsing lives in one
    place (CLAUDE.md code health rule #3).
    """
    return RetrievedChunk(
        content=content,
        video_title=metadata.get("video_title", ""),
        timestamp_start=metadata.get("timestamp_start", 0.0),
        timestamp_end=metadata.get("timestamp_end", 0.0),
        source_id=metadata.get("source_id", ""),
        distance=distance,
        score=score,
        sequence_index=_parse_seq_index(chroma_id),
        seg_start=metadata.get("seg_start"),
        seg_end=metadata.get("seg_end"),
    )


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


class ONNXMultilingualEmbedding:
    """ChromaDB-compatible embedding function using a multilingual ONNX model.

    Mirrors the ONNXCrossEncoder pattern from rerank.py:
    - onnxruntime + tokenizers only (no torch / sentence-transformers)
    - Mean-pooled ONNX forward pass (no query instruction needed)
    - Downloads model files to ~/.bibilab/models/embedding/
    """

    def name(self) -> str:
        return "onnx_multilingual_embedding"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed query strings. Same as __call__ for this model."""
        return self(input)

    def __init__(self) -> None:
        import numpy as np  # noqa: PLC0415

        ensure(EMBEDDING_SPEC_ID)
        model_dir = _embedding_model_dir()
        import onnxruntime as ort  # noqa: PLC0415

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_dir / "onnx" / "model.onnx"),
            providers=ort.get_available_providers(),
            sess_options=so,
        )
        from tokenizers import Tokenizer  # noqa: PLC0415

        self._tokenizer = Tokenizer.from_file(str(model_dir / "onnx" / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=512)

        # pad_token_id from BERT config (0), not tokenizer's <pad> id (1)
        self._pad_id = 0
        self._np = np

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Encode texts to embedding vectors. Mean-pooled."""
        if not input:
            return []

        encoded = [self._tokenizer.encode(t) for t in input]
        max_len = max(len(e.ids) for e in encoded)
        pad_id = self._pad_id

        batch_ids = []
        batch_mask = []
        batch_type_ids = []
        for enc in encoded:
            pad_len = max_len - len(enc.ids)
            batch_ids.append(enc.ids + [pad_id] * pad_len)
            batch_mask.append(enc.attention_mask + [0] * pad_len)
            batch_type_ids.append(enc.type_ids + [0] * pad_len)

        onnx_input = {
            "input_ids": self._np.array(batch_ids, dtype=self._np.int64),
            "attention_mask": self._np.array(batch_mask, dtype=self._np.int64),
            "token_type_ids": self._np.array(batch_type_ids, dtype=self._np.int64),
        }

        # Mean pooling: (batch, seq, hidden) → (batch, hidden)
        hidden = self._session.run(None, onnx_input)[0]
        mask = self._np.array(batch_mask, dtype=self._np.float32)
        masked = hidden * mask[:, :, self._np.newaxis]
        summed = masked.sum(axis=1)
        counts = mask.sum(axis=1, keepdims=True)
        embeddings = summed / counts

        return [emb.tolist() for emb in embeddings]


def _embedding_model_dir() -> Path:
    return models_dir("embedding")


def _default_embedding_function() -> ONNXMultilingualEmbedding:
    return ONNXMultilingualEmbedding()


# Chroma collection name. The string is the on-disk collection name, so
# changing it orphans existing vectors and is a data-migration event, not
# a refactor.
_TRANSCRIPT_COLLECTION = "bibilab_transcripts"


def _get_collection() -> "chromadb.Collection":
    if _TRANSCRIPT_COLLECTION not in _chroma_collections:
        with _chroma_lock:
            if _TRANSCRIPT_COLLECTION not in _chroma_collections:
                import chromadb  # noqa: PLC0415

                client = chromadb.PersistentClient(path=str(bibilab_home() / "chroma"))
                _chroma_collections[_TRANSCRIPT_COLLECTION] = client.get_or_create_collection(
                    _TRANSCRIPT_COLLECTION,
                    embedding_function=_default_embedding_function(),
                )
    return _chroma_collections[_TRANSCRIPT_COLLECTION]


def embed_chunks(
    chunks: list[RagChunk],
    source_id: str,
    meta: VideoMeta,
    list_id: str,
) -> None:
    if not chunks:
        return

    collection = _get_collection()

    clear_embeddings_for_source(source_id)

    ids = [f"{source_id}_{chunk.sequence_index}" for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source_id": source_id,
            "list_id": list_id,
            "video_title": meta.title,
            "timestamp_start": chunk.timestamp_start,
            "timestamp_end": chunk.timestamp_end,
            "sequence_index": chunk.sequence_index,
            "seg_start": chunk.seg_start,
            "seg_end": chunk.seg_end,
        }
        for chunk in chunks
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Embedded %d chunks for source %s", len(chunks), source_id)

    populate_fts(chunks, source_id, meta)


def populate_fts(chunks: list[RagChunk], source_id: str, meta: VideoMeta) -> None:
    """Replace the FTS5 index rows for a source (sync, called from embed thread).

    Clears the source's existing rows first so a re-embed replaces rather than
    appends — mirrors the chroma clear in embed_chunks. Without it a rerun, retry,
    or re-ingest double-indexes the source (every chunk twice), inflating BM25
    document frequency and contaminating retrieval.
    """
    if not chunks:
        return
    clear_fts_for_source_sync(source_id)
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            "INSERT INTO chunks_fts "
            "(content, pinyin, source_id, video_title, timestamp_start, timestamp_end, chunk_id, seg_start, seg_end) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    _tokenize_cjk(chunk.text),
                    _pinyin_index_tokens(chunk.text),
                    source_id,
                    meta.title,
                    chunk.timestamp_start,
                    chunk.timestamp_end,
                    f"{source_id}_{chunk.sequence_index}",
                    chunk.seg_start,
                    chunk.seg_end,
                )
                for chunk in chunks
            ],
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("FTS indexed %d chunks for source %s", len(chunks), source_id)


def clear_fts_for_source_sync(source_id: str) -> None:
    """Delete all FTS rows for a source (sync, for use from worker threads)."""
    db_path = get_db_path()
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM chunks_fts WHERE source_id = ?", (source_id,))
        conn.commit()
    finally:
        conn.close()


def clear_embeddings_for_list(list_id: str) -> None:
    """Delete all ChromaDB chunks belonging to the given list."""
    collection = _get_collection()
    try:
        collection.delete(where={"list_id": list_id})
    except Exception as exc:
        logger.warning("Failed to delete embeddings for list %s: %s", list_id, exc)


def clear_embeddings_for_source(source_id: str) -> None:
    """Delete all ChromaDB chunks belonging to the given source."""
    chroma_path = bibilab_home() / "chroma"
    if not chroma_path.exists():
        return
    collection = _get_collection()
    try:
        collection.delete(where={"source_id": source_id})
    except Exception as exc:
        logger.warning("Failed to delete embeddings for source %s: %s", source_id, exc)


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

    def _sync_query() -> dict:
        collection = _get_collection()
        return collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where={"source_id": {"$in": source_ids}},
        )

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(get_chat_pool(), _sync_query)
    except Exception as exc:  # noqa: BLE001 - ChromaDB errors vary by version
        logger.warning("ChromaDB query failed: %s", exc)
        return []

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]
    ids = results.get("ids") or [[]]

    if not documents[0]:
        return []

    floor = cfg.rag.max_distance

    # ChromaDB returns results sorted by distance ascending; preserve order
    return [
        _row_from_chroma(
            content=documents[0][i],
            metadata=metadatas[0][i],
            distance=distances[0][i],
            chroma_id=ids[0][i] if ids and ids[0] else "",
        )
        for i in range(len(documents[0]))
        if distances[0][i] <= floor
    ]


def _fetch_raw_documents(chunk_ids: list[str]) -> dict[str, str]:
    """Map chunk_id → raw chunk text from Chroma (the canonical chunk-text store).

    The FTS index column holds tokenized text (unigram+bigram soup) needed for
    CJK BM25 matching, not readable prose. Chroma stores the raw chunk.text under
    the same id, so the BM25 arm looks its display/rerank text up there.
    """
    collection = _get_collection()
    got = collection.get(ids=chunk_ids)
    return dict(zip(got.get("ids") or [], got.get("documents") or []))


async def query_fts(
    query_text: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    top_k: int = 30,
) -> list[RetrievedChunk]:
    """Query FTS5 index for transcript chunks matching the query, filtered by source.

    Args:
        query_text: The query string to search for.
        source_ids: List of source UUIDs to filter by.
        cfg: BibilabConfig instance.
        top_k: Maximum number of chunks to return.

    Returns list of RetrievedChunk with score set to the negated FTS5 BM25 rank
    (positive, higher = more relevant) and distance set to 0.0. _chunk_score()
    negates the score to recover "lower = more relevant" ordering.
    """
    if not source_ids:
        return []

    rows = await query_fts_rows(query_text, source_ids, top_k)
    if not rows:
        return []

    # The FTS content column is tokenized token-soup (needed for CJK BM25
    # matching) — it must never reach the reranker or the rendered result.
    # Replace it with the raw chunk text from Chroma (same chunk_id). On a
    # Chroma error, fall back to the FTS content so the BM25 arm still
    # contributes (mirrors query_chunks' fail-soft on Chroma errors).
    chunk_ids = [row["chunk_id"] for row in rows]
    try:
        loop = asyncio.get_running_loop()
        raw_by_id = await loop.run_in_executor(get_chat_pool(), _fetch_raw_documents, chunk_ids)
    except Exception as exc:  # noqa: BLE001 - ChromaDB errors vary by version
        logger.warning("FTS raw-text backfill from Chroma failed: %s", exc)
        raw_by_id = {}

    return [
        _row_from_chroma(
            content=raw_by_id.get(row["chunk_id"]) or row["content"],
            metadata={
                "source_id": row["source_id"],
                "video_title": row["video_title"],
                "timestamp_start": float(row["timestamp_start"]),
                "timestamp_end": float(row["timestamp_end"]),
                "seg_start": row["seg_start"],
                "seg_end": row["seg_end"],
            },
            distance=0.0,
            chroma_id=row["chunk_id"],
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
    if not source_ids:
        return []

    vec_task = query_chunks(query_text, source_ids, cfg, top_k=effective_top_k)
    fts_task = query_fts(query_text, source_ids, cfg, top_k=effective_top_k)

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
    top_k: int,
    scoped_source_ids: list[str] | None = None,
) -> RetrievalResult:
    """Recall-biased locator: hybrid search → rerank → top_k by rerank order.

    No relevance gate (rerank is ordering, not authority) and no per-source
    diversity cap. The LLM filters relevance and decides whether to escalate
    to read_section.
    """
    sources_total = len(source_ids)
    search_pool = scoped_source_ids if scoped_source_ids is not None else source_ids
    effective_top_k = min(max(sources_total * 3, top_k, 10), 60)

    if cfg.rag.hybrid_enabled:
        chunks = await hybrid_search(query_text, search_pool, cfg, effective_top_k=effective_top_k)
    else:
        chunks = await query_chunks(query_text, search_pool, cfg, top_k=effective_top_k)

    candidates_evaluated = len(chunks)

    reranked = False
    if cfg.rag.reranking_enabled and chunks:
        from bibilab.pipeline.rerank import rerank  # noqa: PLC0415

        try:
            chunks = await rerank(query_text, chunks, top_k=len(chunks))
            reranked = True
        except Exception as exc:  # noqa: BLE001 - model load can fail in many ways
            logger.warning("Reranking failed: %s", exc)

    result_chunks = chunks[:top_k]

    final_src_counts: dict[str, int] = {}
    for c in result_chunks:
        final_src_counts[c.source_id] = final_src_counts.get(c.source_id, 0) + 1
    logger.info(
        "retrieve: candidates=%d returned=%d per_source=%s effective_top_k=%d top_k=%d",
        candidates_evaluated,
        len(result_chunks),
        final_src_counts,
        effective_top_k,
        top_k,
    )

    # First-occurrence-per-source (preserves rerank order, single pass).
    first_per_source: dict[str, RetrievedChunk] = {}
    for c in result_chunks:
        first_per_source.setdefault(c.source_id, c)
    source_ids_in_result = list(first_per_source)
    source_coverage = [
        SourceHit(
            source_id=sid,
            video_title=first.video_title,
            best_score=_chunk_score(first),
        )
        for sid, first in first_per_source.items()
    ]

    return RetrievalResult(
        chunks=result_chunks,
        candidates_evaluated=candidates_evaluated,
        sources_with_hits=len(source_ids_in_result),
        sources_total=sources_total,
        source_coverage=source_coverage,
        reranked=reranked,
    )
