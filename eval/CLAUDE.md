# eval/ — RAG Answer Evaluation Framework

CLI tool for evaluating Bibilab RAG pipeline quality. Separate package at repo root.

## Commands

```bash
cd eval && uv sync
uv run bibilab-eval --help
uv run bibilab-eval create <list-id>        # generate eval set + TUI review
uv run bibilab-eval run <eval-set-id>       # run locked cases against test model
uv run bibilab-eval grade <run-id>          # LLM-as-judge grading
uv run bibilab-eval report <run-id>         # aggregate + per-question report
uv run bibilab-eval config                  # TUI to edit profiles
uv run bibilab-eval list                    # list eval sets and runs
uv run pytest                               # run tests
```

## Architecture

```
src/eval/
  cli.py        click entry point, all subcommands
  models.py     EvalCase, EvalSet, EvalRun, RunCaseResult, GradeResult, GradedRun
  config.py    Flat schema: three profiles (generate/test/grade, null=backend) + language (zh|en)
  generate.py   LLM-supervised eval set generation per category
  tui.py        textual TUIs: ReviewApp (cases), ConfigApp (profiles), ReportApp (report)
  runner.py     run locked cases through full chat pipeline with test model
  grader.py     LLM-as-judge: 3 calls per case (context relevance, groundedness, answer relevance)
  reporter.py   aggregate scores, diff, JSON export
  dashboard.py  rich.live TaskDashboard (per-task spinner + sub-status + elapsed)
  storage.py    JSON persistence under ~/.bibilab/evals/{list_id}/
```

## Conventions

- Imports `bibilab.*` via editable install (configured in pyproject.toml `[tool.uv.sources]`)
- `resolve_profile(name)` returns an `AIConfig` — null profile entry falls back to backend `~/.bibilab/config.json`
- `get_response_language()` maps language code → display name ("Chinese"/"English"); single source of truth for output language
- Storage is JSON files under `~/.bibilab/evals/{list_id}/` — portable, git-diffable
- Async functions use `asyncio.run()` at CLI boundaries
- `_call_llm` for sync LLM calls (generation, grading); `stream_with_tools` for chat (runner)
- Long-running batches wrap work in `TaskDashboard` context manager for live progress
- TUI keybinding conventions (shared): ↑/↓ select, ←/→ context switch, enter open/edit, space toggle, ctrl+s save, q quit (confirm if dirty), esc cancel/back
- Tests use `monkeypatch.setattr("eval.storage.bibilab_home", ...)` for isolated temp directories
