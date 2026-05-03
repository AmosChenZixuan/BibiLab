"""Check whether the DB has enough data to start RAG evaluation (#220 / #234).

Usage:
    uv run python scripts/eval/readiness_check.py
    uv run python scripts/eval/readiness_check.py --db /path/to/bibilab.db
"""

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path.home() / ".bibilab" / "bibilab.db"

REQUIRED_QUERIES = 30
REQUIRED_LISTS = 2
REQUIRED_PER_TYPE = 5


def check(db_path: Path) -> dict:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    total = con.execute("SELECT COUNT(*) AS n FROM query_classifications").fetchone()["n"]
    by_type = {
        row["query_type"]: row["n"]
        for row in con.execute(
            "SELECT query_type, COUNT(*) AS n FROM query_classifications GROUP BY query_type"
        ).fetchall()
    }
    lists = con.execute("SELECT COUNT(DISTINCT list_id) AS n FROM query_classifications").fetchone()["n"]

    # Also check if there's actual content to query against
    source_count = con.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"]
    con.close()

    return {
        "total": total,
        "by_type": by_type,
        "lists": lists,
        "sources": source_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RAG eval readiness")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to bibilab.db")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}")
        return 1

    d = check(args.db)

    issues = []

    print(f"query_classifications: {d['total']} rows (need {REQUIRED_QUERIES})")
    if d["total"] < REQUIRED_QUERIES:
        issues.append(f"  -> need {REQUIRED_QUERIES - d['total']} more queries")

    print(f"distinct lists:       {d['lists']} (need {REQUIRED_LISTS})")
    if d["lists"] < REQUIRED_LISTS:
        issues.append(f"  -> need {REQUIRED_LISTS - d['lists']} more lists with queries")

    print(f"type distribution:    {d['by_type']}")
    for qtype in ("factual", "breadth", "analytical"):
        n = d["by_type"].get(qtype, 0)
        if n < REQUIRED_PER_TYPE:
            issues.append(f"  -> '{qtype}' has {n} queries (need {REQUIRED_PER_TYPE})")

    print(f"sources (content):    {d['sources']}")
    if d["sources"] == 0:
        issues.append("  -> no sources ingested — RAG has nothing to retrieve")

    print()

    if issues:
        print("NOT READY:")
        for issue in issues:
            print(issue)
        print()
        missing_types = [t for t in ("factual", "breadth", "analytical") if d["by_type"].get(t, 0) < REQUIRED_PER_TYPE]
        if missing_types:
            print("Hint: try asking different kinds of questions:")
            print('  factual    — "What did the speaker say about X?"')
            print('  analytical — "Compare A and B" / "Why did X happen?"')
            print('  breadth    — "Give me an overview of this video"')
        return 1

    print("READY — eval can start (#220)")
    if d["total"] >= 50:
        print("(also enough data for #234 classification audit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
