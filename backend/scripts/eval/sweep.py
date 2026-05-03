"""Param sweep for RAG retrieval evaluation (#220 / #234).

Runs param combos over the query set defined in queries.yaml. Two modes:

  Full sweep (default) — runs every combo, produces a union chunk set per query
  ready for manual relevance labeling:

    uv run python scripts/eval/sweep.py --run-id sweep-001

  Single combo — runs one param set, outputs per-query JSON + summary.json
  for classification audit (#234):

    uv run python scripts/eval/sweep.py --run-id run-001 --single-combo rrf60_fnone

Filter to specific queries with --queries:

    uv run python scripts/eval/sweep.py --run-id smoke --queries H1,G1 --single-combo rrf60_fnone
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import build_combos, chunk_key, load_queries, load_sweep_config, resolve_list_id

from bibilab.config import get_config
from bibilab.db import get_sources_for_list, log_query_classification
from bibilab.models._enums import QUERY_TYPE_FACTUAL, map_type_to_mode
from bibilab.pipeline.embed import RetrievedChunk, retrieve
from bibilab.pipeline.route import classify_query, params_for_type


def _chunk_to_label_entry(chunk: RetrievedChunk) -> dict:
    return {
        "video_title": chunk.video_title,
        "video_id": chunk.video_id,
        "timestamp_start": chunk.timestamp_start,
        "timestamp_end": chunk.timestamp_end,
        "content": chunk.content,
        "relevant": None,
    }


def _chunk_to_dict(chunk: RetrievedChunk) -> dict:
    return {
        "video_title": chunk.video_title,
        "video_id": chunk.video_id,
        "timestamp_start": chunk.timestamp_start,
        "timestamp_end": chunk.timestamp_end,
        "content": chunk.content,
        "score": chunk.score,
        "distance": chunk.distance,
    }


async def _run_sweep(run_id: str, query_filter: set[str] | None) -> None:
    """Full sweep: all combos × all queries → labels.json."""
    queries = load_queries()
    if query_filter:
        queries = [q for q in queries if q["id"] in query_filter]

    cfg = get_config()
    combos = build_combos(load_sweep_config())
    out_dir = Path.home() / ".bibilab" / "eval" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_data: dict[str, dict] = {}
    total = len(queries) * len(combos)
    n = 0

    for query in queries:
        qid = query["id"]
        list_id = await resolve_list_id(query["list"])
        source_rows = await get_sources_for_list(list_id)
        source_ids = [r["id"] for r in source_rows]

        try:
            qtype = await classify_query(query["text"], cfg)
        except Exception:
            qtype = QUERY_TYPE_FACTUAL
        params = params_for_type(qtype, len(source_ids))

        union: dict[str, dict] = {}

        for combo in combos:
            n += 1
            print(f"[{n}/{total}] {qid} @ {combo['label']}...", end=" ", flush=True)

            result = await retrieve(
                query["text"],
                source_ids,
                cfg,
                params,
                rerank_min_score_override=combo["floor"],
                rrf_k_override=combo["rrf_k"],
            )

            print(f"{len(result.chunks)} chunks", flush=True)

            for chunk in result.chunks:
                key = chunk_key(chunk)
                if key not in union:
                    union[key] = _chunk_to_label_entry(chunk)

            combo_key = combo["label"]
            labels_data.setdefault("combos", {}).setdefault(combo_key, {})
            labels_data["combos"][combo_key][qid] = [chunk_key(c) for c in result.chunks]

        labels_data[qid] = {
            "query_text": query["text"],
            "list": query["list"],
            "list_id": list_id,
            "expected_type": query.get("expected_type"),
            "classified_type": qtype,
            "params": {"depth_per_source": params.depth_per_source, "top_k": params.top_k},
            "chunks": list(union.values()),
        }

        print(f"  → union: {len(union)} unique chunks across {len(combos)} combos")

    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nLabels written to {labels_path}")
    print("Next: open review.html to label chunks, then run score.py")


async def _run_single(run_id: str, combo_label: str, routing_enabled: bool, query_filter: set[str] | None) -> None:
    """Single combo: one param set × all queries → per-query JSON + summary.json."""
    queries = load_queries()
    if query_filter:
        queries = [q for q in queries if q["id"] in query_filter]

    combos = build_combos(load_sweep_config())
    combo = next((c for c in combos if c["label"] == combo_label), None)
    if combo is None:
        valid = ", ".join(c["label"] for c in combos)
        print(f"Unknown combo {combo_label!r}. Valid: {valid}")
        return

    out_dir = Path.home() / ".bibilab" / "eval" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, query in enumerate(queries):
        qid = query["id"]
        list_id = await resolve_list_id(query["list"])
        source_rows = await get_sources_for_list(list_id)
        source_ids = [r["id"] for r in source_rows]

        if routing_enabled:
            cfg = get_config()
            try:
                qtype = await classify_query(query["text"], cfg)
            except Exception:
                qtype = QUERY_TYPE_FACTUAL
        else:
            qtype = QUERY_TYPE_FACTUAL

        params = params_for_type(qtype, len(source_ids))
        effective_mode = map_type_to_mode(qtype)

        try:
            await log_query_classification(
                list_id=list_id,
                query_text=query["text"],
                query_type=qtype,
                effective_mode=effective_mode,
            )
        except Exception:
            pass

        cfg = get_config()
        result = await retrieve(
            query_text=query["text"],
            source_ids=source_ids,
            cfg=cfg,
            params=params,
            rerank_min_score_override=combo["floor"],
            rrf_k_override=combo["rrf_k"],
        )

        record = {
            "id": qid,
            "query_text": query["text"],
            "list": query["list"],
            "list_id": list_id,
            "expected_type": query.get("expected_type"),
            "classified_type": qtype,
            "effective_mode": effective_mode,
            "degenerate": query.get("degenerate", False),
            "params": {"depth_per_source": params.depth_per_source, "top_k": params.top_k},
            "result": {
                "chunks": [_chunk_to_dict(c) for c in result.chunks],
                "candidates_evaluated": result.candidates_evaluated,
                "sources_with_hits": result.sources_with_hits,
                "sources_total": result.sources_total,
                "source_coverage": [
                    {"video_id": s.video_id, "video_title": s.video_title, "best_score": s.best_score}
                    for s in result.source_coverage
                ],
            },
        }
        results.append(record)

        n_chunks = len(result.chunks)
        src_hits = result.sources_with_hits
        src_total = result.sources_total
        print(
            f"[{i + 1}/{len(queries)}] {qid}: {query['text'][:60]}... "
            f"→ {qtype} ({n_chunks} chunks, {src_hits}/{src_total} sources)"
        )

    for r in results:
        (out_dir / f"{r['id']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "combo": combo_label,
        "routing_enabled": routing_enabled,
        "total_queries": len(results),
        "by_classified_type": {},
        "by_expected_type": {},
        "classification_mismatches": [],
        "degenerate_results": [],
    }
    for r in results:
        ct = r["classified_type"]
        summary["by_classified_type"][ct] = summary["by_classified_type"].get(ct, 0) + 1
        et = r["expected_type"] or "boundary"
        summary["by_expected_type"][et] = summary["by_expected_type"].get(et, 0) + 1

        if r["expected_type"] and ct != r["expected_type"]:
            summary["classification_mismatches"].append(
                {
                    "id": r["id"],
                    "expected": r["expected_type"],
                    "actual": ct,
                    "query": r["query_text"],
                }
            )

        if r["degenerate"]:
            summary["degenerate_results"].append(
                {
                    "id": r["id"],
                    "classified_type": ct,
                    "sources_with_hits": r["result"]["sources_with_hits"],
                    "chunks_returned": len(r["result"]["chunks"]),
                }
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Results in {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG eval param sweep")
    parser.add_argument("--run-id", default="sweep-001", help="Output subdirectory under ~/.bibilab/eval/")
    parser.add_argument("--single-combo", default=None, help="Run single combo instead of sweep (e.g. rrf60_fnone)")
    parser.add_argument("--routing-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--queries", default=None, help="Comma-separated query IDs to run (e.g. H1,G1)")
    args = parser.parse_args()

    query_filter = set(args.queries.split(",")) if args.queries else None

    if args.single_combo:
        asyncio.run(_run_single(args.run_id, args.single_combo, args.routing_enabled, query_filter))
    else:
        asyncio.run(_run_sweep(args.run_id, query_filter))


if __name__ == "__main__":
    main()
