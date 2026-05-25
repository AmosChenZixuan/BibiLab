from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from eval.config import resolve_profile, get_language, load_eval_config, save_eval_config
from eval.storage import (
    save_eval_set,
    load_eval_set,
    list_eval_sets,
    list_runs as list_runs_storage,
    load_eval_run,
    load_graded_run,
    export_skeleton,
)
from eval.models import EvalSet


@click.group()
def main():
    """Bibilab RAG Eval Framework — evaluate your RAG pipeline quality."""


@main.command()
@click.argument("list_id")
@click.option(
    "--categories",
    default="narrow,broad,cross_ref,ambiguous,absence,temporal",
    help="Comma-separated categories to generate.",
)
@click.option("--count", default=3, help="Questions per category.")
def create(list_id, categories, count):
    """Generate eval set for a list."""
    from eval.generate import generate_eval_set

    cats = [c.strip() for c in categories.split(",") if c.strip()]

    try:
        ai_cfg = resolve_profile("generate")
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    language = get_language()
    click.echo(f"Generating eval set for list {list_id} ({len(cats)} categories, {count} each, lang={language})...")

    try:
        es = asyncio.run(generate_eval_set(list_id, cats, count, ai_cfg, language))
    except Exception as e:
        click.echo(f"Error generating eval set: {e}", err=True)
        sys.exit(1)

    save_eval_set(es)
    click.echo(f"Eval set saved: {es.id}")
    click.echo(f"  {len(es.cases)} cases generated ({len([c for c in es.cases if c.locked])} locked)")

    from eval.tui import run_review_tui
    run_review_tui(es.id)


@main.command()
@click.argument("eval_set_id")
def review(eval_set_id):
    """Resume interactive review for an existing eval set."""
    from eval.tui import run_review_tui
    run_review_tui(eval_set_id)


@main.command()
@click.argument("eval_set_id")
@click.option("--model", default=None, help="Override test model.")
@click.option("--concurrency", default=None, type=int, help="Max parallel cases (default 4).")
def run(eval_set_id, model, concurrency):
    """Run locked cases against test model."""
    from eval.runner import run_eval, DEFAULT_CONCURRENCY

    try:
        ai_cfg = resolve_profile("test")
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if model:
        ai_cfg.model = model

    conc = concurrency if concurrency is not None else DEFAULT_CONCURRENCY
    click.echo(f"Running eval with model: {ai_cfg.model} (lang={get_language()}, concurrency={conc})...")

    try:
        run_result = asyncio.run(run_eval(eval_set_id, ai_cfg, concurrency=conc))
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


@main.command()
@click.argument("run_id")
def grade(run_id):
    """Grade a run using the grade profile LLM."""
    from eval.grader import grade_run
    from eval.reporter import aggregate_scores, format_report_text

    try:
        ai_cfg = resolve_profile("grade")
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    language = get_language()
    click.echo(f"Grading run {run_id} with {ai_cfg.model} (lang={language})...")

    try:
        gr = asyncio.run(grade_run(run_id, ai_cfg, language))
    except Exception as e:
        click.echo(f"Error grading run: {e}", err=True)
        sys.exit(1)

    click.echo(f"Grading complete. {len(gr.grades)} cases graded.")

    eval_run = load_eval_run(run_id)
    es = load_eval_set(eval_run.eval_set_id)
    cat_map = {c.id: c.category for c in es.cases}
    agg = aggregate_scores(gr.grades, cat_map)
    click.echo()
    click.echo(
        format_report_text(
            es.id, len(gr.grades),
            eval_run.test_profile.get("model", "?"),
            ai_cfg.model, agg,
        )
    )


@main.command()
@click.argument("run_id")
@click.option("--compare", default=None, help="Previous run ID to diff against (use with --json or pre-load in TUI).")
@click.option("--json", "json_out", is_flag=True, help="Machine-readable JSON output (skip TUI).")
def report(run_id, compare, json_out):
    """View eval report (TUI by default; --json for machine output)."""
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
    test_model = eval_run.test_profile.get("model", "?")
    grade_model = gr.grade_profile.get("model", "?")

    diff = None
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

    click.echo(report_json(run_id, es.id, test_model, grade_model, agg, diff, gr, compare))


@main.command()
def config():
    """TUI to edit eval profiles."""
    from eval.tui import run_config_tui
    run_config_tui()


@main.command()
def list():
    """List eval sets and recent runs."""
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
            except Exception:
                pass
            status = f"graded ({gr.grade_profile.get('model', '?')})" if gr else "not graded"
            click.echo(f"  {r.id} — {r.timestamp} ({r.test_profile.get('model', '?')}) — {status}")


@main.command("export-skeleton")
@click.argument("eval_set_id")
@click.option("--target-list", required=True, help="Target list ID.")
def export_skeleton_cmd(eval_set_id, target_list):
    """Export question-only skeleton for a new list."""
    try:
        skeleton = export_skeleton(eval_set_id, target_list)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    save_eval_set(skeleton)
    click.echo(f"Skeleton saved: {skeleton.id} (target list: {target_list})")
    click.echo(f"  {len(skeleton.cases)} questions, all unlocked.")
