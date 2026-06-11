from eval.reporter import aggregate_scores, diff_scores, format_report_text
from eval.models import GradeResult


def make_grade(case_id, cr=4, g=4, ar=4):
    return GradeResult(
        case_id=case_id,
        context_relevance=cr,
        context_relevance_reasoning="ok",
        groundedness=g,
        groundedness_reasoning="ok",
        answer_relevance=ar,
        answer_relevance_reasoning="ok",
    )


def test_aggregate_scores_by_category():
    grades = [
        make_grade("n1", cr=4, g=3, ar=5),
        make_grade("n2", cr=5, g=4, ar=4),
    ]
    categories = {"n1": "single_fact", "n2": "single_fact"}
    agg = aggregate_scores(grades, categories)
    assert agg["single_fact"]["context_relevance"] == 4.5
    assert agg["single_fact"]["groundedness"] == 3.5


def test_aggregate_overall():
    grades = [
        make_grade("n1", cr=4, g=3, ar=5),
        make_grade("b1", cr=2, g=2, ar=3),
    ]
    categories = {"n1": "single_fact", "b1": "enumeration"}
    agg = aggregate_scores(grades, categories)
    assert agg["overall"]["context_relevance"] == 3.0


def test_diff_scores():
    current = {"single_fact": {"context_relevance": 4.0, "groundedness": 3.0, "answer_relevance": 4.0}}
    previous = {"single_fact": {"context_relevance": 3.5, "groundedness": 3.0, "answer_relevance": 4.5}}
    diff = diff_scores(current, previous)
    assert diff["single_fact"]["context_relevance"] == 0.5
    assert diff["single_fact"]["groundedness"] == 0.0
    assert diff["single_fact"]["answer_relevance"] == -0.5


def test_diff_scores_missing_category():
    current = {"single_fact": {"context_relevance": 4.0, "groundedness": 3.0, "answer_relevance": 4.0}}
    previous = {}
    diff = diff_scores(current, previous)
    assert "single_fact" not in diff


def test_format_report_text():
    agg = {
        "overall": {"context_relevance": 4.0, "groundedness": 3.5, "answer_relevance": 3.8},
        "single_fact": {"context_relevance": 4.5, "groundedness": 4.0, "answer_relevance": 4.0},
    }
    text = format_report_text("my-list", 3, "glm-4.7-flash", "gpt-4o", agg, None)
    assert "my-list" in text
    assert "glm-4.7-flash" in text
    assert "gpt-4o" in text
    assert "4.5" in text


def test_aggregate_ignores_failed_grades():
    grades = [
        make_grade("n1", cr=4, g=None, ar=3),  # g=None means grading failed
        make_grade("n2", cr=5, g=4, ar=4),
    ]
    categories = {"n1": "single_fact", "n2": "single_fact"}
    agg = aggregate_scores(grades, categories)
    assert agg["single_fact"]["groundedness"] == 4.0  # only n2 counted


def test_count_failed_grades():
    from eval.reporter import count_failed_grades
    grades = [
        make_grade("n1", cr=4, g=None, ar=3),
        make_grade("n2", cr=None, g=4, ar=None),
    ]
    failed = count_failed_grades(grades)
    assert failed == {"context_relevance": 1, "groundedness": 1, "answer_relevance": 1}
