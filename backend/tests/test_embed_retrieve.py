"""Integration tests for retrieve() neighbor-pull (issue #333)."""

from __future__ import annotations

import uuid

import chromadb
import pytest

from bibilab.config import AIConfig, BackendConfig, BibilabConfig, RagConfig
from bibilab.pipeline import embed
from bibilab.pipeline.embed import (
    RetrievalParams,
    RetrievedChunk,
    _chunk_score,
    _chunks_from_chroma_get,
    _parse_seq_index,
    _row_from_chroma,
    retrieve,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _cfg(*, threshold: int = 2, reranking: bool = True) -> BibilabConfig:
    return BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
        rag=RagConfig(
            neighbor_scarcity_threshold=threshold,
            reranking_enabled=reranking,
            hybrid_enabled=False,  # vector-only path: simpler test surface
            max_distance=10.0,
        ),
    )


def _build_collection(video_id: str = "v1", n_chunks: int = 6):
    """Build an in-memory Chroma collection with `n_chunks` sequential chunks."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"test_{video_id}_{uuid.uuid4().hex[:8]}")
    collection.add(
        ids=[f"{video_id}_{i}" for i in range(n_chunks)],
        documents=[f"chunk {i} content" for i in range(n_chunks)],
        metadatas=[
            {
                "video_id": video_id,
                "video_title": f"Video {video_id}",
                "timestamp_start": float(i * 30),
                "timestamp_end": float((i + 1) * 30),
                "sequence_index": i,
            }
            for i in range(n_chunks)
        ],
    )
    return collection


@pytest.fixture
def patch_chroma(monkeypatch):
    """Install a Chroma collection + identity video_id resolution."""

    def install(collection):
        monkeypatch.setattr(embed, "_get_collection", lambda cfg: collection)

        async def _fake_resolve(source_ids, passed):
            return passed if passed is not None else list(source_ids)

        monkeypatch.setattr(embed, "_resolve_video_ids", _fake_resolve)

    return install


@pytest.fixture
def patch_rerank(monkeypatch):
    """Patch rerank() to score chunks via a per-test mapping (chroma_id -> score)."""
    from bibilab.pipeline import rerank as rerank_mod

    score_map: dict[str, float] = {}

    async def _fake_rerank(query, chunks, top_k):
        for c in chunks:
            cid = f"{c.video_id}_{c.sequence_index}" if c.sequence_index is not None else c.video_id
            c.score = score_map.get(cid, -10.0)
        chunks.sort(key=lambda c: -c.score if c.score is not None else c.distance)
        return chunks[:top_k]

    monkeypatch.setattr(rerank_mod, "rerank", _fake_rerank)
    return score_map


def _params(top_k: int = 8, depth: int = 2, expected_hits: str = "few") -> RetrievalParams:
    return RetrievalParams(top_k=top_k, depth_per_source=depth, expected_hits=expected_hits)


# ---------------------------------------------------------------------------
# Unit tests — pure parsing / helpers
# ---------------------------------------------------------------------------


class TestRetrievedChunkSequenceIndex:
    """`sequence_index` placed as LAST dataclass field; positional-arg compat
    matters for legacy `RetrievedChunk(...)` callers in other tests."""

    def test_positional_construction_with_sequence_index(self):
        c = RetrievedChunk("content", "title", 0.0, 10.0, "vid", 0.5, None, 7)
        assert c.sequence_index == 7
        assert c.is_neighbor is False


class TestParseSeqIndex:
    def test_simple(self):
        assert _parse_seq_index("BV1abc_460") == 460

    def test_video_id_with_underscore(self):
        # rsplit on _ takes the trailing integer
        assert _parse_seq_index("my_video_id_42") == 42

    def test_malformed(self):
        assert _parse_seq_index("no_underscore_suffix_x") is None
        assert _parse_seq_index("") is None


class TestRowFromChroma:
    def test_basic(self):
        c = _row_from_chroma(
            content="hello",
            metadata={
                "video_id": "v1",
                "video_title": "T",
                "timestamp_start": 1.0,
                "timestamp_end": 2.0,
            },
            distance=0.4,
            chroma_id="v1_5",
        )
        assert c.content == "hello"
        assert c.video_id == "v1"
        assert c.sequence_index == 5
        assert c.is_neighbor is False
        assert c.score is None

    def test_neighbor_flag(self):
        c = _row_from_chroma(
            content="",
            metadata={"video_id": "v"},
            distance=float("inf"),
            chroma_id="v_3",
            is_neighbor=True,
        )
        assert c.is_neighbor is True


class TestChunksFromChromaGet:
    """Parses the FLAT shape returned by collection.get (NOT collection.query)."""

    def test_empty(self):
        assert _chunks_from_chroma_get({}) == []
        assert _chunks_from_chroma_get({"ids": []}) == []

    def test_flat_shape(self):
        rows = {
            "ids": ["v1_0", "v1_2"],
            "documents": ["d0", "d2"],
            "metadatas": [
                {"video_id": "v1", "video_title": "T", "timestamp_start": 0.0, "timestamp_end": 10.0},
                {"video_id": "v1", "video_title": "T", "timestamp_start": 20.0, "timestamp_end": 30.0},
            ],
        }
        chunks = _chunks_from_chroma_get(rows, is_neighbor=True)
        assert len(chunks) == 2
        assert [c.sequence_index for c in chunks] == [0, 2]
        assert all(c.is_neighbor for c in chunks)
        assert all(c.distance == float("inf") for c in chunks)
        assert all(c.score is None for c in chunks)


class TestChunkScoreSentinel:
    def test_neighbor_chunk_score_is_inf(self):
        c = RetrievedChunk("", "", 0.0, 0.0, "v", 0.1, score=5.0, is_neighbor=True)
        # is_neighbor short-circuits regardless of score/distance.
        assert _chunk_score(c) == float("inf")

    def test_hit_chunk_score_uses_score(self):
        c = RetrievedChunk("", "", 0.0, 0.0, "v", 0.1, score=5.0)
        assert _chunk_score(c) == -5.0


# ---------------------------------------------------------------------------
# Integration tests — drive real retrieve() via EphemeralClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bug_repro_single_hit_pulls_both_neighbors(patch_chroma, patch_rerank):
    """[#333 bug repro] Hit on chunk 2 alone → result = {1, 2, 3} ascending."""
    collection = _build_collection("v1", n_chunks=6)
    patch_chroma(collection)
    patch_rerank["v1_2"] = 1.0  # Only chunk_2 survives the gate.

    result = await retrieve("q", ["v1"], _cfg(threshold=2), _params())

    assert result.neighbors_pulled == 2
    assert [c.sequence_index for c in result.chunks] == [1, 2, 3]
    assert [c.is_neighbor for c in result.chunks] == [True, False, True]
    # Neighbor sentinel.
    for c in result.chunks:
        if c.is_neighbor:
            assert c.score is None
            assert c.distance == float("inf")


@pytest.mark.asyncio
async def test_two_clustered_hits_dedupe_no_duplicates(patch_chroma, patch_rerank):
    """Adjacent hits {2, 3} → neighbors {1, 4}; the dedup bug (Critical #2)
    would have re-injected 2/3 as duplicates here."""
    collection = _build_collection("v1", n_chunks=6)
    patch_chroma(collection)
    patch_rerank["v1_2"] = 1.0
    patch_rerank["v1_3"] = 0.9

    result = await retrieve("q", ["v1"], _cfg(threshold=2), _params())

    seq = [c.sequence_index for c in result.chunks]
    assert seq == [1, 2, 3, 4], f"expected [1,2,3,4], got {seq}"
    assert result.neighbors_pulled == 2  # {1, 4}, NOT 4 (which would mean 2/3 duplicated)
    # No duplicate sequence_index anywhere.
    assert len(seq) == len(set(seq))


@pytest.mark.asyncio
async def test_lower_clamp_hit_at_index_zero(patch_chroma, patch_rerank):
    """Hit at sequence_index=0 → no negative neighbor; only +1 pulled."""
    collection = _build_collection("v1", n_chunks=6)
    patch_chroma(collection)
    patch_rerank["v1_0"] = 1.0

    result = await retrieve("q", ["v1"], _cfg(threshold=2), _params())

    assert [c.sequence_index for c in result.chunks] == [0, 1]
    assert result.neighbors_pulled == 1


@pytest.mark.asyncio
async def test_upper_clamp_missing_id_silently_dropped(patch_chroma, patch_rerank):
    """Hit at last sequence_index → only the existing neighbor returned;
    Chroma `get` omits missing ids without error."""
    collection = _build_collection("v1", n_chunks=6)  # ids: v1_0..v1_5
    patch_chroma(collection)
    patch_rerank["v1_5"] = 1.0  # Last existing chunk.

    result = await retrieve("q", ["v1"], _cfg(threshold=2), _params())

    # v1_6 does not exist; only v1_4 returns.
    assert [c.sequence_index for c in result.chunks] == [4, 5]
    assert result.neighbors_pulled == 1


@pytest.mark.asyncio
async def test_n_above_threshold_no_neighbor_pull(patch_chroma, patch_rerank):
    """N=3 hits with threshold=2 → byte-identical to today; no neighbors pulled."""
    collection = _build_collection("v1", n_chunks=6)
    patch_chroma(collection)
    patch_rerank["v1_1"] = 1.0
    patch_rerank["v1_2"] = 0.9
    patch_rerank["v1_3"] = 0.8

    # depth_per_source=3 so all three hits survive the per-source cap.
    result = await retrieve("q", ["v1"], _cfg(threshold=2), _params(depth=3))

    assert result.neighbors_pulled == 0
    assert [c.sequence_index for c in result.chunks] == [1, 2, 3]
    assert all(not c.is_neighbor for c in result.chunks)


@pytest.mark.asyncio
async def test_threshold_zero_bypasses(patch_chroma, patch_rerank):
    """cfg.neighbor_scarcity_threshold=0 → no neighbors regardless of N."""
    collection = _build_collection("v1", n_chunks=6)
    patch_chroma(collection)
    patch_rerank["v1_2"] = 1.0

    result = await retrieve("q", ["v1"], _cfg(threshold=0), _params())

    assert result.neighbors_pulled == 0
    assert [c.sequence_index for c in result.chunks] == [2]


@pytest.mark.asyncio
async def test_reranked_false_bypasses(patch_chroma, monkeypatch):
    """reranking_enabled=False → gate didn't run → no neighbor pull."""
    collection = _build_collection("v1", n_chunks=6)
    patch_chroma(collection)

    result = await retrieve("q", ["v1"], _cfg(threshold=2, reranking=False), _params())

    assert result.neighbors_pulled == 0
    assert result.reranked is False


@pytest.mark.asyncio
async def test_cross_source_neighbors_in_both(patch_chroma, patch_rerank, monkeypatch):
    """Two sources × 1 hit each → ±1 neighbors fetched in both."""
    # Single shared collection holding both videos' chunks.
    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"two_sources_{uuid.uuid4().hex[:8]}")
    for vid in ("v1", "v2"):
        collection.add(
            ids=[f"{vid}_{i}" for i in range(4)],
            documents=[f"{vid} chunk {i}" for i in range(4)],
            metadatas=[
                {
                    "video_id": vid,
                    "video_title": f"Video {vid}",
                    "timestamp_start": float(i * 30),
                    "timestamp_end": float((i + 1) * 30),
                    "sequence_index": i,
                }
                for i in range(4)
            ],
        )
    monkeypatch.setattr(embed, "_get_collection", lambda cfg: collection)

    async def _fake_resolve(source_ids, passed):
        return passed if passed is not None else list(source_ids)

    monkeypatch.setattr(embed, "_resolve_video_ids", _fake_resolve)

    patch_rerank["v1_1"] = 1.0
    patch_rerank["v2_2"] = 0.95
    # Bump max_distance: ChromaDB returns L2 distances; defaults reject most fakes.
    result = await retrieve("q", ["v1", "v2"], _cfg(threshold=2), _params())

    by_source = {}
    for c in result.chunks:
        by_source.setdefault(c.video_id, []).append(c.sequence_index)
    assert sorted(by_source["v1"]) == [0, 1, 2], by_source
    assert sorted(by_source["v2"]) == [1, 2, 3], by_source
    assert result.neighbors_pulled == 4
    # source_coverage best_score should use the original hit, not the neighbor sentinel.
    for sh in result.source_coverage:
        assert sh.best_score != float("inf")


@pytest.mark.asyncio
async def test_end_to_end_neighbors_pulled_propagates_to_chat_metadata(patch_chroma, patch_rerank, monkeypatch):
    """`metadata.rag.calls[0].neighbors_pulled` is populated through execute_retrieve."""
    from bibilab.pipeline import chat_tools

    collection = _build_collection("v1", n_chunks=6)
    patch_chroma(collection)
    patch_rerank["v1_2"] = 1.0

    # execute_retrieve returns a dict (the tool_result payload).
    result_dict = await chat_tools.execute_retrieve(
        query="q",
        source_ids=["v1"],
        cfg=_cfg(threshold=2),
        source_map={"v1": "src-1"},
    )
    assert result_dict.get("neighbors_pulled") == 2
