# eval/ — RAG Answer Evaluation Framework

CLI tool for evaluating Bibilab RAG pipeline quality. Separate package at repo root — a **thin HTTP client** of a running backend (`backend_url` in `~/.bibilab/eval_config.json`, default `http://127.0.0.1:8765`); no `bibilab` dependency, so its venv stays ~31 MB and it can target a remote backend.

## Commands

```bash
cd eval && uv sync
uv run bibilab-eval --help
uv run bibilab-eval create <list-id>        # generate eval set + TUI review
uv run bibilab-eval review <eval-set-id>    # resume TUI review for an existing set
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
  api.py        the only module that talks to the backend: lists/sources/transcript reads, POST /api/eval/llm (bare LLM), POST /api/eval/run_chat (chat turn); unwraps {"detail": {"error": code}} into RuntimeError
  cli.py        click entry point, all subcommands
  models.py     EvalCase, EvalSet, EvalRun, RunCaseResult, GradeResult, GradedRun
  config.py    Flat schema: three profiles (generate/test/grade, null=backend) + language (zh|en) + backend_url
  generate.py   LLM-supervised eval set generation per category
  tui.py        textual TUIs: ReviewApp (cases), ConfigApp (profiles + backend_url), ReportApp (report)
  runner.py     run locked cases via POST /api/eval/run_chat; map_response builds RunCaseResult (llm_context ← response llm_context, the exact LLM-bound tool messages)
  grader.py     LLM-as-judge: 3 calls per case (context relevance, groundedness, answer relevance)
  reporter.py   aggregate scores, diff, JSON export
  dashboard.py  rich.live TaskDashboard (per-task spinner + sub-status + elapsed)
  storage.py    JSON persistence under ~/.bibilab/evals/{list_id}/
```

## Conventions

- The target backend must serve `/api/*` — the Docker image and any SPA-built source run do; a bare `python -m bibilab.main` without a `web/dist` build mounts routes at the root only, so every eval call 404s.
- Never `import bibilab` — all backend data and every LLM call goes through `api.py` over HTTP. LLM calls route through `POST /api/eval/llm` so provider requests are byte-identical to the backend's own (`_call_llm` runs server-side).
- `resolve_profile(name)` returns `ProfileSnapshot | None` — None means "no `llm` override in the request", the backend serves the call with its own configured LLM. Empty-string profile fields also count as unset (a blank api_key must inherit the backend's key, not override it).
- `get_response_language()` returns the language code ("zh"/"en") — sent as the `language` request field, which the grounding prompt keys on (not the display name)
- Storage is JSON files under `~/.bibilab/evals/{list_id}/` — portable, git-diffable, host-side (the one thing not behind HTTP)
- Async functions use `asyncio.run()` at CLI boundaries
- Long-running batches wrap work in `TaskDashboard` context manager for live progress
- TUI keybinding conventions (shared): ↑/↓ select, ←/→ context switch, enter open/edit, space toggle, ctrl+s save, q quit (confirm if dirty), esc cancel/back
- Tests use `monkeypatch.setattr("eval.storage.bibilab_home", ...)` for temp dirs and assign `httpx.MockTransport` to `api.transport` / `api.async_transport` for HTTP seams
