from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from bibilab.config import bibilab_home

from eval.models import EvalSet, EvalRun, GradedRun, EvalCase


def _evals_root() -> Path:
    d = bibilab_home() / "evals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _evals_dir_read(eval_set_id: str) -> Path:
    return _evals_root() / eval_set_id


def _evals_dir_write(eval_set_id: str) -> Path:
    d = _evals_root() / eval_set_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


# -- Eval Sets -----------------------------------------------------------

def save_eval_set(eval_set: EvalSet) -> None:
    d = _evals_dir_write(eval_set.id)
    _atomic_write(d / "eval_set.json", eval_set.model_dump_json(indent=2))


def load_eval_set(eval_set_id: str) -> EvalSet:
    path = _evals_dir_read(eval_set_id) / "eval_set.json"
    if not path.exists():
        raise FileNotFoundError(f"Eval set '{eval_set_id}' not found")
    return EvalSet.model_validate_json(path.read_text())


def list_eval_sets() -> list[tuple[str, str]]:
    """Return [(eval_set_id, list_id), ...] for all eval sets."""
    result: list[tuple[str, str]] = []
    root = _evals_root()
    for entry in root.iterdir():
        if entry.is_dir():
            candidate = entry / "eval_set.json"
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text())
                    result.append((data["id"], data["list_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return result


# -- Runs ----------------------------------------------------------------

def save_eval_run(run: EvalRun) -> None:
    d = _evals_dir_write(run.eval_set_id) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    _atomic_write(d / f"{run.id}.json", run.model_dump_json(indent=2))


def load_eval_run(run_id: str) -> EvalRun:
    root = _evals_root()
    for entry in root.iterdir():
        if entry.is_dir():
            candidate = entry / "runs" / f"{run_id}.json"
            if candidate.exists():
                return EvalRun.model_validate_json(candidate.read_text())
    raise FileNotFoundError(f"Run '{run_id}' not found")


def list_runs(eval_set_id: str) -> list[EvalRun]:
    d = _evals_dir_read(eval_set_id) / "runs"
    if not d.exists():
        return []
    runs: list[EvalRun] = []
    for f in sorted(d.glob("*.json")):
        runs.append(EvalRun.model_validate_json(f.read_text()))
    return runs


# -- Grades --------------------------------------------------------------

def save_graded_run(gr: GradedRun) -> None:
    run = load_eval_run(gr.run_id)
    d = _evals_dir_write(run.eval_set_id) / "grades"
    d.mkdir(parents=True, exist_ok=True)
    _atomic_write(d / f"{gr.run_id}.json", gr.model_dump_json(indent=2))


def load_graded_run(run_id: str) -> GradedRun:
    root = _evals_root()
    for entry in root.iterdir():
        if entry.is_dir():
            candidate = entry / "grades" / f"{run_id}.json"
            if candidate.exists():
                return GradedRun.model_validate_json(candidate.read_text())
    raise FileNotFoundError(f"Graded run '{run_id}' not found")


# -- Export --------------------------------------------------------------

def export_skeleton(eval_set_id: str, target_list_id: str) -> EvalSet:
    original = load_eval_set(eval_set_id)
    skeleton_cases = []
    for case in original.cases:
        skeleton_cases.append(
            EvalCase(
                id=str(uuid.uuid4()),
                category=case.category,
                question=case.question,
                expected_answer_draft="",
                expected_sources=[],
                locked=False,
                notes="",
            )
        )
    return EvalSet(
        id=str(uuid.uuid4()),
        list_id=target_list_id,
        created_at=original.created_at,
        updated_at=original.created_at,
        cases=skeleton_cases,
    )
