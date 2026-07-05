from __future__ import annotations

import asyncio
import sys

import click

from eval.config import resolve_profile, get_response_language
from eval.models import DEFAULT_CATEGORIES, DEFAULT_FLOOR
from eval.storage import (
    save_eval_set,
    load_eval_set,
    list_eval_sets,
    list_runs as list_runs_storage,
    load_eval_run,
    load_graded_run,
)


# ── interactive helpers ──────────────────────────────────────────────


def _pick_list() -> str | None:
    from eval import api

    try:
        rows = api.get_lists()
    except Exception as e:
        click.echo(f"Error: backend unreachable: {e}", err=True)
        return None
    if not rows:
        click.echo("No lists found. Import videos first.")
        return None
    click.echo("\nLists:")
    for i, row in enumerate(rows, 1):
        click.echo(f"  [{i}] {row['name']} ({row['source_count']} sources)")
    click.echo("  [0] Back")
    try:
        choice = click.prompt("Pick list", type=int)
    except click.Abort:
        return None
    if choice == 0:
        return None
    if 1 <= choice <= len(rows):
        return rows[choice - 1]["id"]
    click.echo("Invalid choice.")
    return None


def _pick_eval_set(purpose: str = "select") -> str | None:
    sets = list_eval_sets()
    if not sets:
        click.echo("No eval sets found. Run 'create' first.")
        return None
    click.echo(f"\nEval sets ({purpose}):")
    for i, (es_id, list_id) in enumerate(sets, 1):
        try:
            es = load_eval_set(es_id)
            locked = len(es.locked_cases)
            click.echo(f"  [{i}] {es_id[:8]}... — list {list_id[:8]}..., {len(es.cases)} cases, {locked} locked")
        except Exception:
            click.echo(f"  [{i}] {es_id[:8]}... — list {list_id[:8]}...")
    click.echo("  [0] Back")
    try:
        choice = click.prompt("Pick eval set", type=int)
    except click.Abort:
        return None
    if choice == 0:
        return None
    if 1 <= choice <= len(sets):
        return sets[choice - 1][0]
    click.echo("Invalid choice.")
    return None


def _pick_run(graded: bool = False) -> str | None:
    sets = list_eval_sets()
    candidates: list[tuple[str, str, str, str]] = []  # (run_id, ts, test_model, grade_model)
    for es_id, _list_id in sets:
        for r in list_runs_storage(es_id):
            has_grade = True
            try:
                gr = load_graded_run(r.id)
                grade_model = gr.grade_profile.model
            except FileNotFoundError:
                has_grade = False
                grade_model = ""
            if graded and has_grade:
                candidates.append((r.id, r.timestamp, r.test_profile.model, grade_model))
            elif not graded and not has_grade:
                candidates.append((r.id, r.timestamp, r.test_profile.model, grade_model))

    label = "Graded runs" if graded else "Ungraded runs"
    if not candidates:
        click.echo(f"No {label.lower()} found.")
        return None
    click.echo(f"\n{label}:")
    for i, (rid, ts, test_model, grade_model) in enumerate(candidates, 1):
        extra = f" grade: {grade_model}" if grade_model else ""
        click.echo(f"  [{i}] {rid[:8]}... — test: {test_model}{extra} ({ts})")
    click.echo("  [0] Back")
    try:
        choice = click.prompt("Pick run", type=int)
    except click.Abort:
        return None
    if choice == 0:
        return None
    if 1 <= choice <= len(candidates):
        return candidates[choice - 1][0]
    click.echo("Invalid choice.")
    return None


def _top_menu():
    click.echo("\nBibilab Eval")
    click.echo("  [1] Create  — generate eval set from a bibilab list")
    click.echo("  [2] Review  — review existing eval set")
    click.echo("  [3] Run     — run locked cases against test model")
    click.echo("  [4] Grade   — LLM-as-judge grading")
    click.echo("  [5] Report  — view graded report")
    click.echo("  [6] Config  — edit eval profiles")
    click.echo("  [7] List    — list eval sets and runs")
    try:
        choice = click.prompt("Pick", type=click.IntRange(1, 7))
    except click.Abort:
        click.echo("")
        return

    if choice == 1:
        _create_interactive()
    elif choice == 2:
        _review_interactive()
    elif choice == 3:
        _run_interactive()
    elif choice == 4:
        _grade_interactive()
    elif choice == 5:
        _report_interactive()
    elif choice == 6:
        from eval.tui import run_config_tui
        run_config_tui()
    elif choice == 7:
        _list_cmd()


def _create_interactive():
    list_id = _pick_list()
    if list_id is None:
        return

    cats_default = DEFAULT_CATEGORIES
    cats_input = click.prompt("Categories (comma-separated)", default=cats_default, show_default=False)
    click.echo(f"  Categories: {cats_input}")
    cats = [c.strip() for c in cats_input.split(",") if c.strip()]
    floor = click.prompt("Floor (min questions per category)", type=int, default=DEFAULT_FLOOR)
    _do_create(list_id, cats, floor)


def _review_interactive():
    es_id = _pick_eval_set("review")
    if es_id is None:
        return
    from eval.tui import run_review_tui
    run_review_tui(es_id)


def _run_interactive():
    es_id = _pick_eval_set("run")
    if es_id is None:
        return
    _do_run(es_id, model=None, concurrency=None)


def _grade_interactive():
    run_id = _pick_run(graded=False)
    if run_id is None:
        return
    _do_grade(run_id)


def _report_interactive():
    run_id = _pick_run(graded=True)
    if run_id is None:
        return
    from eval.tui import run_report_tui
    run_report_tui(run_id)


# ── CLI ──────────────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """Bibilab RAG Eval Framework — evaluate your RAG pipeline quality."""
    if ctx.invoked_subcommand is None:
        _top_menu()


# ── shared execution ─────────────────────────────────────────────────


def _llm_override(profile) -> dict | None:
    """ProfileSnapshot → request `llm` override; None profile = no override
    (the backend serves the call with its own configured LLM). Empty strings
    count as unset too — a blank api_key/model must inherit the backend's
    value, not override it with ""."""
    if profile is None:
        return None
    return {k: v for k, v in profile.model_dump().items() if v not in (None, "")} or None


def _profile_label(profile) -> str:
    return profile.model if profile else "(backend default)"


def _do_create(list_id: str, cats: list[str], floor: int):
    from eval import api
    from eval.generate import generate_eval_set, resolve_counts

    try:
        profile = resolve_profile("generate")
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    language = get_response_language()
    counts = resolve_counts(cats, floor)
    total = sum(counts.values())
    click.echo(
        f"Generating eval set for list {list_id} "
        f"({len(cats)} categories, floor {floor}, {total} questions, lang={language})..."
    )

    try:
        all_sources = api.get_sources(list_id)
        es = generate_eval_set(list_id, all_sources, counts, _llm_override(profile), language)
    except Exception as e:
        click.echo(f"Error generating eval set: {e}", err=True)
        sys.exit(1)

    save_eval_set(es)
    click.echo(f"Eval set saved: {es.id}")
    click.echo(f"  {len(es.cases)} cases generated ({len([c for c in es.cases if c.locked])} locked)")

    from eval.tui import run_review_tui
    run_review_tui(es.id)


def _do_run(eval_set_id: str, model: str | None, concurrency: int | None):
    from eval.runner import run_eval, DEFAULT_CONCURRENCY

    try:
        profile = resolve_profile("test")
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    llm = _llm_override(profile)
    if model:
        llm = {**(llm or {}), "model": model}

    conc = concurrency if concurrency is not None else DEFAULT_CONCURRENCY
    label = model or _profile_label(profile)
    click.echo(f"Running eval with model: {label} (lang={get_response_language()}, concurrency={conc})...")

    try:
        run_result = asyncio.run(run_eval(eval_set_id, llm, concurrency=conc))
    except Exception as e:
        click.echo(f"Error running eval: {e}", err=True)
        sys.exit(1)

    errors = [c for c in run_result.cases if c.error]
    ok = len(run_result.cases) - len(errors)
    click.echo(f"Run complete: {run_result.id}")
    click.echo(f"  {ok} passed, {len(errors)} errors")
    if errors:
        for e in errors:
            click.echo(f"  ✗ {e.case_id}: {e.error}")
    click.echo(f"  Grade it: bibilab-eval grade {run_result.id}")


def _do_grade(run_id: str):
    from eval.grader import grade_run
    from eval.reporter import aggregate_scores, format_report_text, count_failed_grades

    try:
        profile = resolve_profile("grade")
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    language = get_response_language()
    click.echo(f"Grading run {run_id} with {_profile_label(profile)} (lang={language})...")

    try:
        gr = asyncio.run(grade_run(run_id, _llm_override(profile), language))
    except Exception as e:
        click.echo(f"Error grading run: {e}", err=True)
        sys.exit(1)

    click.echo(f"Grading complete. {len(gr.grades)} cases graded.")

    eval_run = load_eval_run(run_id)
    es = load_eval_set(eval_run.eval_set_id)
    cat_map = {c.id: c.category for c in es.cases}
    agg = aggregate_scores(gr.grades, cat_map)
    failed = count_failed_grades(gr.grades)
    click.echo()
    click.echo(
        format_report_text(
            es.id, len(gr.grades),
            eval_run.test_profile.model,
            gr.grade_profile.model, agg,
        )
    )
    if any(failed.values()):
        click.echo()
        click.echo(
            f"  ⚠ failed grades: CR={failed['context_relevance']} "
            f"G={failed['groundedness']} AR={failed['answer_relevance']}"
        )


def _list_cmd():
    sets = list_eval_sets()
    if not sets:
        click.echo("No eval sets found.")
        return

    for es_id, list_id in sets:
        try:
            es = load_eval_set(es_id)
        except Exception as e:
            click.echo(f"{es_id} (list: {list_id}) — error loading: {e}")
            continue

        locked = len(es.locked_cases)
        click.echo(f"{es_id} (list: {list_id}, {len(es.cases)} cases, {locked} locked)")
        runs = list_runs_storage(es_id)
        for r in runs:
            gr = None
            try:
                gr = load_graded_run(r.id)
            except FileNotFoundError:
                pass
            status = f"graded ({gr.grade_profile.model})" if gr else "not graded"
            click.echo(f"  {r.id} — {r.timestamp} ({r.test_profile.model}) — {status}")


# ── commands ─────────────────────────────────────────────────────────


@main.command()
@click.argument("list_id", required=False, default=None)
@click.option("--categories", default=DEFAULT_CATEGORIES, help="Comma-separated categories.")
@click.option("--floor", default=DEFAULT_FLOOR, help="Minimum questions per category (failure-prone types get a surplus on top).")
def create(list_id, categories, floor):
    """Generate eval set for a list."""
    if list_id is None:
        list_id = _pick_list()
        if list_id is None:
            return
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    _do_create(list_id, cats, floor)


@main.command()
@click.argument("eval_set_id", required=False, default=None)
def review(eval_set_id):
    """Resume interactive review for an existing eval set."""
    if eval_set_id is None:
        eval_set_id = _pick_eval_set("review")
        if eval_set_id is None:
            return
    from eval.tui import run_review_tui
    run_review_tui(eval_set_id)


@main.command()
@click.argument("eval_set_id", required=False, default=None)
@click.option("--model", default=None, help="Override test model.")
@click.option("--concurrency", default=None, type=int, help="Max parallel cases.")
def run(eval_set_id, model, concurrency):
    """Run locked cases against test model."""
    if eval_set_id is None:
        eval_set_id = _pick_eval_set("run")
        if eval_set_id is None:
            return
    _do_run(eval_set_id, model, concurrency)


@main.command()
@click.argument("run_id", required=False, default=None)
def grade(run_id):
    """Grade a run using the grade profile LLM."""
    if run_id is None:
        run_id = _pick_run(graded=False)
        if run_id is None:
            return
    _do_grade(run_id)


@main.command()
@click.argument("run_id", required=False, default=None)
@click.option("--compare", default=None, help="Previous run ID to diff against.")
@click.option("--json", "json_out", is_flag=True, help="Machine-readable JSON output (skip TUI).")
def report(run_id, compare, json_out):
    """View eval report (TUI by default; --json for machine output)."""
    if run_id is None:
        run_id = _pick_run(graded=True)
        if run_id is None:
            return

    try:
        gr = load_graded_run(run_id)
        eval_run = load_eval_run(run_id)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not json_out:
        from eval.tui import run_report_tui
        run_report_tui(run_id, compare)
        return

    from eval.reporter import aggregate_scores, diff_scores, report_json

    es = load_eval_set(eval_run.eval_set_id)
    cat_map = {c.id: c.category for c in es.cases}
    agg = aggregate_scores(gr.grades, cat_map)
    test_model = eval_run.test_profile.model
    grade_model = gr.grade_profile.model

    diff = None
    compared_id = compare
    if compare:
        try:
            prev_gr = load_graded_run(compare)
            prev_run = load_eval_run(compare)
            prev_es = load_eval_set(prev_run.eval_set_id)
            prev_cat_map = {c.id: c.category for c in prev_es.cases}
            prev_agg = aggregate_scores(prev_gr.grades, prev_cat_map)
            diff = diff_scores(agg, prev_agg)
        except FileNotFoundError as e:
            click.echo(f"Warning: comparison run not found: {e}", err=True)
            compared_id = None

    click.echo(report_json(run_id, es.id, test_model, grade_model, agg, diff, gr, compared_id))


@main.command()
def config():
    """TUI to edit eval profiles."""
    from eval.tui import run_config_tui
    run_config_tui()


@main.command()
def list():
    """List eval sets and recent runs."""
    _list_cmd()
