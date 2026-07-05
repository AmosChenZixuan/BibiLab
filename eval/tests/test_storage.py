import uuid
import pytest
from eval.storage import (
    save_eval_set,
    load_eval_set,
    list_eval_sets,
    save_eval_run,
    load_eval_run,
    list_runs,
    save_graded_run,
    load_graded_run,
)
from eval.models import EvalCase, EvalSet, EvalRun, RunCaseResult, GradeResult, GradedRun


def _make_eval_set(list_id="list-1", cases=None):
    return EvalSet(
        id=str(uuid.uuid4()),
        list_id=list_id,
        created_at="2026-05-24T10:00:00",
        updated_at="2026-05-24T10:00:00",
        cases=cases or [],
    )


def _make_run(eval_set_id="es-1"):
    return EvalRun(
        id=str(uuid.uuid4()),
        eval_set_id=eval_set_id,
        test_profile={"model": "glm-4.7-flash"},
        timestamp="2026-05-24T15:00:00",
        cases=[],
    )


def test_save_and_load_eval_set(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.storage.bibilab_home", lambda: tmp_path)
    es = _make_eval_set()
    es.cases = [
        EvalCase(
            id="c1",
            category="single_fact",
            question="什么是X？",
            expected_answer_draft="X is...",
            locked=True,
            notes="",
        )
    ]
    save_eval_set(es)
    loaded = load_eval_set(es.id)
    assert loaded.id == es.id
    assert loaded.cases[0].question == "什么是X？"
    assert loaded.cases[0].locked is True


def test_save_and_load_eval_run(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.storage.bibilab_home", lambda: tmp_path)
    es = _make_eval_set()
    save_eval_set(es)

    run = EvalRun(
        id=str(uuid.uuid4()),
        eval_set_id=es.id,
        test_profile={"model": "glm-4.7-flash"},
        timestamp="2026-05-24T15:00:00",
        cases=[
            RunCaseResult(
                case_id="c1",
                answer="X is a technique...",
                citations=[{"index": 1, "source_id": "s1"}],
                rag_calls=[],
                tool_blocks=[],
                llm_duration_ms=1500,
                error=None,
            )
        ],
    )
    save_eval_run(run)
    loaded = load_eval_run(run.id)
    assert loaded.id == run.id
    assert loaded.cases[0].answer == "X is a technique..."


def test_list_eval_sets(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.storage.bibilab_home", lambda: tmp_path)
    es1 = _make_eval_set(list_id="list-a")
    es2 = _make_eval_set(list_id="list-b")
    save_eval_set(es1)
    save_eval_set(es2)
    sets = list_eval_sets()
    assert len(sets) == 2


def test_list_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.storage.bibilab_home", lambda: tmp_path)
    es = _make_eval_set()
    save_eval_set(es)
    run = EvalRun(
        id=str(uuid.uuid4()),
        eval_set_id=es.id,
        test_profile={"model": "glm-4.7-flash"},
        timestamp="2026-05-24T15:00:00",
        cases=[],
    )
    save_eval_run(run)
    runs = list_runs(run.eval_set_id)
    assert len(runs) == 1
    assert runs[0].id == run.id


def test_save_and_load_graded_run(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.storage.bibilab_home", lambda: tmp_path)
    es = _make_eval_set()
    save_eval_set(es)
    run = EvalRun(
        id="run-1", eval_set_id=es.id,
        test_profile={"model": "glm-4.7-flash"},
        timestamp="2026-05-24T15:00:00", cases=[],
    )
    save_eval_run(run)

    gr = GradedRun(
        run_id="run-1",
        grade_profile={"model": "gpt-4o"},
        timestamp="2026-05-24T16:00:00",
        grades=[
            GradeResult(
                case_id="c1",
                context_relevance=4,
                context_relevance_reasoning="good coverage",
                groundedness=5,
                groundedness_reasoning="all cited",
                answer_relevance=4,
                answer_relevance_reasoning="minor omission",
            )
        ],
    )
    save_graded_run(gr)
    loaded = load_graded_run(gr.run_id)
    assert loaded.grades[0].context_relevance == 4


def test_eval_set_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.storage.bibilab_home", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        load_eval_set("nonexistent")
