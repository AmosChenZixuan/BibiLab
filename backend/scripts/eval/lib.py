"""Shared helpers for the RAG eval framework.

Kept small and focused: YAML loading, DB list resolution, chunk key,
and metric functions shared by sweep.py and score.py.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_QUERIES_YAML = Path(__file__).resolve().parent / "queries.yaml"


def load_queries() -> list[dict]:
    """Load query definitions from queries.yaml."""
    with open(_QUERIES_YAML) as f:
        data = yaml.safe_load(f)
    return data["queries"]


def load_sweep_config() -> dict:
    """Load sweep configuration from queries.yaml."""
    with open(_QUERIES_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("sweep", {})


def build_combos(sweep_config: dict) -> list[dict]:
    """Build param combos from sweep config (product of all axes)."""
    rrf_k_values = sweep_config.get("rrf_k", [60])
    floor_values = sweep_config.get("floor", [None])
    return [
        {"rrf_k": k, "floor": f, "label": f"rrf{k}_f{f if f is not None else 'none'}"}
        for k in rrf_k_values
        for f in floor_values
    ]


async def resolve_list_id(name: str) -> str:
    """Resolve a list name to its UUID via the DB."""
    from bibilab.db import get_db

    async with get_db() as db:
        row = await (await db.execute("SELECT id FROM lists WHERE name = ?", (name,))).fetchone()
    if not row:
        raise ValueError(f"No list named {name!r}. Edit queries.yaml or import the list first.")
    return row["id"]


def chunk_key(chunk) -> str:
    """Stable dedup key for a chunk (RetrievedChunk, dict, or labels.json entry)."""
    if isinstance(chunk, dict):
        return f"{chunk.get('video_id', '')}_{chunk.get('timestamp_start', 0)}_{chunk.get('timestamp_end', 0)}"
    return f"{chunk.video_id}_{chunk.timestamp_start}_{chunk.timestamp_end}"


# ── Metrics (used by score.py) ──


def precision_at_k(ranked_relevance: list[bool], k: int) -> float:
    if not ranked_relevance:
        return 0.0
    subset = ranked_relevance[:k]
    return sum(subset) / len(subset)


def recall(ranked_relevance: list[bool], total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(ranked_relevance) / total_relevant


def mrr(ranked_relevance: list[bool]) -> float:
    for i, rel in enumerate(ranked_relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0
