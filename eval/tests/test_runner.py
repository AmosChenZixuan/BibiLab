import importlib
from eval.reporter import _mean
from eval.models import RunCaseResult


def test_mean():
    scores = [4.0, 5.0, 3.0, 4.0, 5.0]
    result = _mean(scores)
    assert result == 4.2


def test_mean_empty():
    assert _mean([]) == 0.0


def test_mean_single():
    assert _mean([3.7]) == 3.7


def test_run_case_result_defaults():
    r = RunCaseResult(
        case_id="c1",
        answer="test",
        citations=[],
        rag_calls=[],
        tool_blocks=[],
        llm_duration_ms=0,
        error=None,
    )
    assert r.error is None
    assert r.case_id == "c1"


def test_runner_module_imports():
    """Smoke test: eval.runner must import without errors.

    Regression guard: the chat-pipeline tool surface was renamed in the
    bounded-sections work (READ_SOURCE_TOOL -> READ_SECTION_TOOL). A stale
    import in eval.runner crashed the entire eval framework at import time.
    """
    module = importlib.import_module("eval.runner")
    assert hasattr(module, "CHAT_TOOLS")
    assert module.CHAT_TOOLS, "CHAT_TOOLS should not be empty"
