"""ChromaDB chunk embedding step."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb

import sqlite3
from pathlib import Path

from bibilab.adapters.base import VideoMeta
from bibilab.config import BibilabConfig, bibilab_home, models_dir
from bibilab.db import _tokenize_cjk, get_db_path, query_fts_rows
from bibilab.model_registry import EMBEDDING_SPEC_ID, _integrity_ok, ensure, get_spec
from bibilab.models._enums import _RELEVANCE_MARGIN_BY_MODE, RetrievalParams
from bibilab.pipeline.chat_inference_pool import get_chat_pool
from bibilab.pipeline.chunk import RagChunk

logger = logging.getLogger(__name__)


_chroma_collections: dict[str, "chromadb.Collection"] = {}


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
    # True for chunks added by neighbor-pull. Excludes them from source_coverage
    # best_score and from any future relevance sort.
    is_neighbor: bool = False


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
    # Telemetry for #277 I-4. dropped_by_gate is 0 when gate did not run
    # (rerank disabled or failed); reranked disambiguates that case.
    dropped_by_gate: int = 0
    reranked: bool = False
    # Actual margin used by _quantile_gate (derived from mode).
    # None when gate did not run (rerank disabled or failed); actual margin when it did.
    gate_margin: float | None = None
    neighbors_pulled: int = 0


def _chunk_score(chunk: RetrievedChunk) -> float:
    """Canonical relevance where lower = more relevant."""
    if chunk.is_neighbor:
        return float("inf")
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
    is_neighbor: bool = False,
) -> RetrievedChunk:
    """Build a RetrievedChunk from one Chroma row (id + document + metadata + distance).

    Shared by query_chunks (vector), query_fts, and neighbor fetch so row
    parsing lives in one place (CLAUDE.md code health rule #3).
    """
    sid = metadata.get("source_id", "")
    if not sid:
        logger.warning("ChromaDB entry %s has empty source_id metadata — legacy data?", chroma_id)

    return RetrievedChunk(
        content=content,
        video_title=metadata.get("video_title", ""),
        timestamp_start=metadata.get("timestamp_start", 0.0),
        timestamp_end=metadata.get("timestamp_end", 0.0),
        source_id=sid,
        distance=distance,
        score=score,
        sequence_index=_parse_seq_index(chroma_id),
        is_neighbor=is_neighbor,
    )


def _chunks_from_chroma_get(results: dict, *, is_neighbor: bool = False) -> list[RetrievedChunk]:
    """Parse ChromaDB collection.get() flat-shape result into RetrievedChunks.

    collection.get returns flat parallel lists (ids, documents, metadatas);
    no distances (id-based fetch carries no relevance score). Missing ids
    are omitted from the result by Chroma — caller treats as end-of-source.
    Chunks whose chroma_id fails to parse are dropped (caller needs
    sequence_index for ordering).
    """
    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    return [
        _row_from_chroma(
            content=doc,
            metadata=md,
            distance=float("inf"),
            chroma_id=cid,
            score=None,
            is_neighbor=is_neighbor,
        )
        for cid, doc, md in zip(ids, documents, metadatas, strict=False)
    ]


def _diverse_top_k(ranked: list[RetrievedChunk], depth: int, k: int) -> list[RetrievedChunk]:
    """Pick up to k chunks, allowing at most depth chunks from any single video.

    Strict cap — no leftover backfill. If the depth cap blocks slots from
    filling, returns fewer than k. Callers must treat short returns as the
    normal case (see #277). Adaptive depth (matching depth to the pool's
    distinct-source count) is the caller's responsibility — compute it
    via _adaptive_depth() and pass the effective depth in.
    """
    per_src: dict[str, int] = {}
    picked: list[RetrievedChunk] = []
    for c in ranked:
        cur = per_src.get(c.source_id, 0)
        if cur < depth:
            picked.append(c)
            per_src[c.source_id] = cur + 1
            if len(picked) == k:
                return picked
    return picked


def _adaptive_depth(spec_depth: int, top_k: int, num_sources_in_pool: int) -> int:
    """Effective per-source depth given the pool's distinct-source count.

    When the LLM (post-#287) scopes retrieval to one source via
    `selected_source_ids`, the static depth from _PARAMS_BY_MODE under-returns.
    This raises depth toward top_k as the pool narrows. Empty pool returns
    spec to avoid divide-by-zero; _diverse_top_k handles empty input fine).
    """
    if num_sources_in_pool <= 0:
        return spec_depth
    return max(spec_depth, math.ceil(top_k / num_sources_in_pool))


def _quantile_gate(chunks: list[RetrievedChunk], margin: float = 2.0) -> list[RetrievedChunk]:
    """Keep chunks whose rerank score clears the relevance threshold.

    threshold = max(median(scores), top - margin)
    - median: when the whole pool is marginal, forces a half-cut (#277).
    - margin: when there is a clear winner, keeps chunks within margin of top.
    No 0 floor — bge logits have no "relevance-neutral" zero point.

    Returns [] when no chunk clears the threshold. The caller treats an empty
    result as "library has no coverage" (#332); never pads with a sub-threshold
    chunk.

    Caller must ensure chunks were reranked (score = bge logit). For
    RRF-domain scores the margin is uncalibrated; gate is bypassed in that
    case (see retrieve()).

    Args:
        chunks: Reranked RetrievedChunk list (score = bge logit).
        margin: bge logit units within which to keep chunks below the top.
            Derived from _RELEVANCE_MARGIN_BY_MODE based on mode.
    """
    if not chunks:
        return chunks
    scores = sorted((c.score for c in chunks if c.score is not None), reverse=True)
    if not scores:
        return chunks
    top = scores[0]
    median = scores[len(scores) // 2]
    threshold = max(median, top - margin)
    kept = [c for c in chunks if c.score is not None and c.score >= threshold]
    return kept


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


def is_embedding_model_downloaded() -> bool:
    return _integrity_ok(get_spec(EMBEDDING_SPEC_ID))


def _default_embedding_function() -> ONNXMultilingualEmbedding:
    return ONNXMultilingualEmbedding()


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
    source_id: str,
    meta: VideoMeta,
    list_id: str,
    cfg: BibilabConfig,
) -> None:
    if not chunks:
        return

    collection = _get_collection(cfg)

    clear_embeddings_for_source(source_id, cfg)

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
    """Insert transcript chunks into the FTS5 index (sync, called from embed thread)."""
    if not chunks:
        return
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            "INSERT INTO chunks_fts "
            "(content, source_id, video_title, timestamp_start, timestamp_end, chunk_id, seg_start, seg_end) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    _tokenize_cjk(chunk.text),
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
    except Exception:
        logger.exception("FTS population failed for source %s", source_id)
        raise
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


def clear_embeddings_for_list(list_id: str, cfg: BibilabConfig) -> None:
    """Delete all ChromaDB chunks belonging to the given list."""
    collection = _get_collection(cfg)
    try:
        collection.delete(where={"list_id": list_id})
    except Exception as exc:
        logger.warning("Failed to delete embeddings for list %s: %s", list_id, exc)


def clear_embeddings_for_source(source_id: str, cfg: BibilabConfig) -> None:
    """Delete all ChromaDB chunks belonging to the given source."""
    chroma_path = bibilab_home() / "chroma"
    if not chroma_path.exists():
        return
    collection = _get_collection(cfg)
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
        collection = _get_collection(cfg)
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

    return [
        _row_from_chroma(
            content=row["content"],
            metadata={
                "source_id": row["source_id"],
                "video_title": row["video_title"],
                "timestamp_start": float(row["timestamp_start"]),
                "timestamp_end": float(row["timestamp_end"]),
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
    params: RetrievalParams,
    scoped_source_ids: list[str] | None = None,
) -> RetrievalResult:
    """High-level retrieval that wraps hybrid search with metadata.

    Selection is driven by RetrievalParams: depth_per_source caps chunks from
    any single video; top_k sets the total returned.

    Args:
        source_ids: the full list of sources (used to derive sources_total for the UI chip).
        scoped_source_ids: optional subset to search within; when selected_source_ids are provided
            by the LLM (via query_list_metadata), this is the filtered pool so search is scoped
            while the UI chip still shows the full list total.
    """
    sources_total = len(source_ids)
    search_pool = scoped_source_ids if scoped_source_ids is not None else source_ids
    effective_top_k = min(max(sources_total * 3, params.top_k, 10), 60)

    if cfg.rag.hybrid_enabled:
        chunks = await hybrid_search(query_text, search_pool, cfg, effective_top_k=effective_top_k)
    else:
        chunks = await query_chunks(query_text, search_pool, cfg, top_k=effective_top_k)

    candidates_evaluated = len(chunks)
    pool_src_counts: dict[str, int] = {}
    for c in chunks:
        pool_src_counts[c.source_id] = pool_src_counts.get(c.source_id, 0) + 1
    logger.info(
        "retrieve post-hybrid: candidates=%d per_source=%s effective_top_k=%d",
        candidates_evaluated,
        pool_src_counts,
        effective_top_k,
    )

    reranked = False
    dropped_by_gate = 0
    gate_margin: float | None = None
    if cfg.rag.reranking_enabled and chunks:
        from bibilab.pipeline.rerank import rerank  # noqa: PLC0415

        try:
            chunks = await rerank(query_text, chunks, top_k=len(chunks))
            reranked = True
        except Exception as exc:  # noqa: BLE001 - model load can fail in many ways
            logger.warning("Reranking failed, gate skipped: %s", exc)

        if reranked:
            pre_gate = len(chunks)
            gate_margin = _RELEVANCE_MARGIN_BY_MODE.get(params.mode, _RELEVANCE_MARGIN_BY_MODE["narrow"])
            chunks = _quantile_gate(chunks, margin=gate_margin)
            dropped_by_gate = pre_gate - len(chunks)
            logger.info(
                "retrieve post-rerank+gate: pre=%d post=%d dropped=%d top_score=%.4f margin=%.1f(mode=%s)",
                pre_gate,
                len(chunks),
                dropped_by_gate,
                chunks[0].score if chunks else -999,
                gate_margin,
                params.mode,
            )

    # candidates_evaluated: pool size (pre-diverse-top-k); used for logs, not UI.
    # adaptive depth ensures #287 scoped queries (1-3 sources) aren't capped at spec.
    num_sources_in_pool = len({c.source_id for c in chunks})
    effective_depth = _adaptive_depth(params.depth_per_source, params.top_k, num_sources_in_pool)
    result_chunks = _diverse_top_k(chunks, effective_depth, params.top_k)

    # Pull ±1 chunks when rerank left few hits so the LLM sees enough
    # surrounding transcript to anchor pronouns and contrastive concepts.
    neighbors_pulled = 0
    threshold = cfg.rag.neighbor_scarcity_threshold
    if threshold > 0 and reranked and len(result_chunks) <= threshold:
        # Hit ids in chroma-id shape {src}_{seq} so subtraction matches the
        # candidate format (any other key shape silently misses the dedup).
        hit_ids = {f"{c.source_id}_{c.sequence_index}" for c in result_chunks if c.sequence_index is not None}
        candidate_ids: list[str] = []
        seen: set[str] = set()
        for c in result_chunks:
            if c.sequence_index is None:
                continue
            for n in (c.sequence_index - 1, c.sequence_index + 1):
                if n < 0:
                    continue
                cid = f"{c.source_id}_{n}"
                if cid in hit_ids or cid in seen:
                    continue
                candidate_ids.append(cid)
                seen.add(cid)

        if candidate_ids:

            def _sync_get() -> dict:
                return _get_collection(cfg).get(ids=candidate_ids)

            try:
                loop = asyncio.get_running_loop()
                neighbor_rows = await loop.run_in_executor(get_chat_pool(), _sync_get)
            except Exception as exc:  # noqa: BLE001 - Chroma errors vary by version
                logger.warning("Neighbor-fetch failed: %s", exc)
                neighbor_rows = {}

            n_chunks = _chunks_from_chroma_get(neighbor_rows, is_neighbor=True)
            # Defensive: a neighbor whose metadata.source_id differs from any
            # current hit's source_id means Chroma metadata is desynced from
            # the chroma_id prefix. Drop with warn so source_coverage never
            # ends up with a neighbor-only entry (StopIteration risk).
            hit_sources = {c.source_id for c in result_chunks}
            unexpected = [c for c in n_chunks if c.source_id not in hit_sources]
            if unexpected:
                logger.warning(
                    "neighbor-pull: dropping %d chunks with unexpected source_id (metadata corruption?)",
                    len(unexpected),
                )
                n_chunks = [c for c in n_chunks if c.source_id in hit_sources]
            if not n_chunks and candidate_ids:
                logger.debug(
                    "neighbor-fetch returned 0 of %d candidate ids (deleted source or stale ids?)",
                    len(candidate_ids),
                )
            if n_chunks:
                # Merge hits + neighbors; sort within each source by
                # sequence_index ascending so the LLM reads continuous
                # transcript under the #297 fence. Inter-source order
                # preserved via the original-hit appearance rank.
                source_rank = {sid: i for i, sid in enumerate(dict.fromkeys(c.source_id for c in result_chunks))}
                result_chunks = sorted(
                    result_chunks + n_chunks,
                    key=lambda c: (source_rank[c.source_id], c.sequence_index),
                )
                neighbors_pulled = len(n_chunks)

    final_src_counts: dict[str, int] = {}
    for c in result_chunks:
        final_src_counts[c.source_id] = final_src_counts.get(c.source_id, 0) + 1
    logger.info(
        "retrieve post-diverse: final=%d per_source=%s depth=%d(adaptive_from=%d) sources_in_pool=%d",
        len(result_chunks),
        final_src_counts,
        effective_depth,
        params.depth_per_source,
        num_sources_in_pool,
    )
    if neighbors_pulled:
        logger.info(
            "retrieve neighbor-pull: hits=%d pulled=%d",
            len(result_chunks) - neighbors_pulled,
            neighbors_pulled,
        )

    # source_coverage derived from result_chunks so the chip reflects what the LLM actually saw.
    # best_score uses the first non-neighbor chunk so neighbor sentinel (+inf) does not
    # poison the source ordering on threshold-met turns.
    source_ids_in_result = list(dict.fromkeys(c.source_id for c in result_chunks))
    source_coverage = [
        SourceHit(
            source_id=sid,
            video_title=next(c.video_title for c in result_chunks if c.source_id == sid),
            best_score=_chunk_score(next(c for c in result_chunks if c.source_id == sid and not c.is_neighbor)),
        )
        for sid in source_ids_in_result
    ]

    return RetrievalResult(
        chunks=result_chunks,
        candidates_evaluated=candidates_evaluated,
        sources_with_hits=len(source_ids_in_result),
        sources_total=sources_total,
        source_coverage=source_coverage,
        dropped_by_gate=dropped_by_gate,
        reranked=reranked,
        gate_margin=gate_margin,
        neighbors_pulled=neighbors_pulled,
    )
