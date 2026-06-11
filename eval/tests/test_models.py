import uuid
from eval.models import EvalCase, EvalSet, RunCaseResult, EvalRun, GradeResult, GradedRun


def test_eval_case_creation():
    case = EvalCase(
        id=str(uuid.uuid4()),
        category="single_fact",
        question="什么是RAG？",
        expected_answer_draft="RAG 是检索增强生成...",
        locked=False,
        notes="",
    )
    assert case.category == "single_fact"
    assert not case.locked


def test_eval_case_invalid_category():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvalCase(
            id="1",
            category="invalid",
            question="x",
            expected_answer_draft="x",
            locked=False,
            notes="",
        )


def test_run_case_result_with_error():
    r = RunCaseResult(
        case_id="c1",
        answer="",
        citations=[],
        rag_calls=[],
        tool_blocks=[],
        llm_duration_ms=0,
        error="model unavailable",
    )
    assert r.error == "model unavailable"


def test_grade_result_range_validation():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GradeResult(
            case_id="c1",
            context_relevance=0,  # rubric is 1-5; 0 invalid
            context_relevance_reasoning="",
            groundedness=3,
            groundedness_reasoning="",
            answer_relevance=3,
            answer_relevance_reasoning="",
        )


def test_grade_result_valid():
    g = GradeResult(
        case_id="c1",
        context_relevance=4,
        context_relevance_reasoning="chunks cover most of the question",
        groundedness=5,
        groundedness_reasoning="all claims sourced",
        answer_relevance=4,
        answer_relevance_reasoning="minor tangent on pricing",
    )
    assert 1 <= g.context_relevance <= 5


def test_eval_set_serialization():
    es = EvalSet(
        id=str(uuid.uuid4()),
        list_id="list-1",
        created_at="2026-05-24T10:00:00",
        updated_at="2026-05-24T10:00:00",
        cases=[],
    )
    d = es.model_dump()
    assert d["list_id"] == "list-1"
    assert d["cases"] == []


def test_eval_set_locked_cases():
    es = EvalSet(
        id=str(uuid.uuid4()),
        list_id="list-1",
        created_at="2026-05-24T10:00:00",
        updated_at="2026-05-24T10:00:00",
        cases=[
            EvalCase(
                id="c1", category="single_fact", question="q1",
                expected_answer_draft="a", locked=True, notes="",
            ),
            EvalCase(
                id="c2", category="enumeration", question="q2",
                expected_answer_draft="a", locked=False, notes="",
            ),
        ],
    )
    assert len(es.locked_cases) == 1
    assert es.locked_cases[0].id == "c1"
