# RAG Retrieval Evaluation Framework

Param-sweep → label → score pipeline for calibrating the retrieval stack
(reranker floor, RRF-k, and future tunables).

## Quick Start

```bash
# All commands run from backend/
cd backend

# 1. Check your environment is ready
uv run python scripts/eval/readiness_check.py

# 2. Run a single-combo smoke test (2 queries, 1 combo)
uv run python scripts/eval/sweep.py --run-id smoke --queries H1,G1 --single-combo rrf60_fnone

# 3. Full sweep (all combos × all queries)
uv run python scripts/eval/sweep.py --run-id sweep-001

# 4. (Optional) LLM-assisted pre-labeling to speed up manual review
uv run python scripts/eval/prelabel.py --run-id sweep-001

# 5. Manual labeling — open review.html in a browser, load labels.json
#    Toggle relevance with keys 1-9, j/k to navigate, Ctrl+S to save.

# 6. Score
uv run python scripts/eval/score.py --run-id sweep-001
uv run python scripts/eval/score.py --run-id sweep-001 --by-list
```

## File Map

```
scripts/eval/
├── README.md
├── queries.yaml          # Query set + sweep config (edit this to add queries)
├── lib.py                # Shared helpers (list resolution, metrics, YAML loading)
├── sweep.py              # Param sweep runner (full sweep or single-combo mode)
├── score.py              # Metric computation (P@K, Recall, MRR; --by-list)
├── prelabel.py           # LLM-assisted relevance pre-labeling (optional)
├── review.html           # Self-contained manual review UI
└── readiness_check.py    # DB readiness check before running eval
```

## When to Re-run

- **Reranker swap** (#225 was an example — score range changes invalidate floor decisions)
- **Major chunking changes** (different candidate pool composition)
- **Tool-calling search** (#228) ships — verify quality didn't regress
- **New lists added** — verify retrieval quality generalizes to new content types

## Query Set Design

30 queries across 3 lists (9+13+16 sources), 3 content types (fiction, recipes, tech).
4 categories: factual (12), breadth (9), analytical (6), boundary (3).
3 degenerate queries (H8, G8, A8) probe metadata-aggregation limits.
Full details in `docs/internal/rag_tuning.md`.

## Adding Queries

Edit `queries.yaml`:

```yaml
queries:
  - id: X1
    list: MyList        # must match a list name in your DB
    text: What is ...?
    expected_type: factual
    scope: single
    degenerate: false
```

List names are resolved against the DB at runtime — no UUIDs needed in the YAML.

## Param Sweep Config

```yaml
sweep:
  rrf_k: [30, 60, 90]
  floor: [null, -2.0, 0.0]
```

All combos (product of axes) are run. Each combo gets its own ranking in labels.json.
