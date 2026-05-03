"""Score labeled eval data with precision@k, recall, and MRR.

Usage:
    uv run python scripts/eval/score.py --run-id sweep-001
    uv run python scripts/eval/score.py --run-id sweep-001 --by-list
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import chunk_key, mrr, precision_at_k, recall


def _compute(
    query_ids: list[str],
    data: dict,
    combos: dict,
    total_relevant: dict[str, int],
) -> list[dict]:
    rows = []
    for combo_key in sorted(combos):
        combo_chunks = combos[combo_key]
        all_p5, all_p10, all_pall, all_recall, all_mrr = [], [], [], [], []

        for qid in query_ids:
            ranked_keys = combo_chunks.get(qid, [])
            relevance_lookup = {chunk_key(c): c["relevant"] for c in data[qid]["chunks"]}
            ranked_rel = [bool(relevance_lookup.get(k)) for k in ranked_keys]

            all_p5.append(precision_at_k(ranked_rel, 5))
            all_p10.append(precision_at_k(ranked_rel, 10))
            all_pall.append(precision_at_k(ranked_rel, len(ranked_rel)) if ranked_rel else 0.0)
            all_recall.append(recall(ranked_rel, total_relevant[qid]))
            all_mrr.append(mrr(ranked_rel))

        n = len(query_ids)
        rows.append(
            {
                "combo": combo_key,
                "p5": sum(all_p5) / n,
                "p10": sum(all_p10) / n,
                "pall": sum(all_pall) / n,
                "recall": sum(all_recall) / n,
                "mrr": sum(all_mrr) / n,
            }
        )
    return rows


def _print_table(rows: list[dict], label: str | None = None) -> str | None:
    """Print a metric table. Returns best combo label."""
    if not rows:
        return None
    if label:
        print(f"\n--- {label} ---")
    print(f"{'Combo':<20} {'P@5':>6} {'P@10':>6} {'P@all':>6} {'Recall':>7} {'MRR':>6}")
    print("-" * 55)
    best = max(rows, key=lambda r: r["mrr"])
    for row in rows:
        marker = " *" if row["combo"] == best["combo"] else ""
        print(
            f"{row['combo']:<20} {row['p5']:>6.3f} {row['p10']:>6.3f} "
            f"{row['pall']:>6.3f} {row['recall']:>7.3f} {row['mrr']:>6.3f}{marker}"
        )
    return best["combo"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score #220 eval labels")
    parser.add_argument("--run-id", default="sweep-001", help="Sweep run subdirectory under ~/.bibilab/eval/")
    parser.add_argument("--by-list", action="store_true", help="Show per-list breakdown")
    args = parser.parse_args()

    labels_path = Path.home() / ".bibilab" / "eval" / args.run_id / "labels.json"
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return

    data = json.loads(labels_path.read_text(encoding="utf-8"))
    combos = data.pop("combos", {})
    query_ids = sorted(k for k in data if k != "combos")

    unlabeled = [qid for qid in query_ids if any(c["relevant"] is None for c in data[qid]["chunks"])]
    if unlabeled:
        print(f"WARNING: {len(unlabeled)} queries have unlabeled chunks: {', '.join(unlabeled)}")
        print("Set 'relevant': true/false for all chunks before scoring.\n")

    total_relevant = {qid: sum(1 for c in data[qid]["chunks"] if c["relevant"] is True) for qid in query_ids}

    # Overall table
    overall = _compute(query_ids, data, combos, total_relevant)
    best = _print_table(overall)
    if best:
        print(f"\nBest overall: {best}  (MRR={max(r['mrr'] for r in overall):.4f})")

    # Per-list breakdown
    if args.by_list:
        queries_by_list: dict[str, list[str]] = defaultdict(list)
        for qid in query_ids:
            queries_by_list[data[qid]["list"]].append(qid)

        for list_name, qids in sorted(queries_by_list.items()):
            n_rel = sum(total_relevant[q] for q in qids)
            rows = _compute(qids, data, combos, total_relevant)
            _print_table(rows, f"{list_name} ({len(qids)} queries, {n_rel} relevant)")


if __name__ == "__main__":
    main()
