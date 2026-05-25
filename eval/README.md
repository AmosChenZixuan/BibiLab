# Bibilab RAG Eval Framework

Evaluate your RAG pipeline quality. Generate eval sets from real list content, run tests against a less-capable local model, and grade answers with LLM-as-judge.

## Setup

```bash
cd eval
uv sync
```

## Configure

```bash
uv run bibilab-eval config
```

Three profiles:
- **generate** — LLM that creates eval questions (defaults to backend config)
- **test** — less-capable model for robustness testing (custom: ollama glm-4.7-flash)
- **grade** — LLM that judges answer quality (defaults to backend config)

Each profile can inherit from `~/.bibilab/config.json` or be configured independently.

## Workflow

```bash
# 1. Generate eval set from a list with processed sources
uv run bibilab-eval create <list-id>
# → LLM generates questions → TUI opens → review, edit, lock cases → save

# 1b. Resume review later (if you quit before finishing)
uv run bibilab-eval review <eval-set-id>

# 2. Run locked cases against test model
uv run bibilab-eval run <eval-set-id>
# → full chat pipeline per question → capture answers + telemetry

# 3. Grade the run
uv run bibilab-eval grade <run-id>
# → 3 LLM calls per case → scores 0-5 for context relevance, groundedness, answer relevance

# 4. View report
uv run bibilab-eval report <run-id>
uv run bibilab-eval report <run-id> --compare <prev-run-id>  # diff
uv run bibilab-eval report <run-id> --json                    # machine-readable

# 5. Bootstrap another list
uv run bibilab-eval export-skeleton <eval-set-id> --target-list <other-list-id>
```

## Eval Categories

| Category | What it tests |
|---|---|
| narrow | Precision — answer exists in one chunk |
| broad | Recall — answers across many sources |
| cross_ref | Multi-source synthesis |
| ambiguous | Context resolution |
| absence | Proving a topic is NOT in sources |
| temporal | Timestamp reasoning across versions |

## RAG Triad

- **Context relevance**: Do retrieved chunks contain the answer?
- **Groundedness**: Are all claims in the answer backed by chunks?
- **Answer relevance**: Does the answer directly address the question?

## Storage

All data under `~/.bibilab/evals/{list_id}/`:
- `eval_set.json` — cases with lock status
- `runs/{run_id}.json` — answers + citations + telemetry
- `grades/{run_id}.json` — per-case 0-5 scores + reasoning
