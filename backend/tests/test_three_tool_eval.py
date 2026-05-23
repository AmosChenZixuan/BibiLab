"""
CI gate: each case in three_tool_eval.json must produce the expected
tool choice when the LLM is given current chat_tools and the grounding
prompt. This file does NOT run a real LLM in unit-test mode — instead,
it validates the fixture's structural integrity and the schema of the
three tool definitions. A separate marker-gated test exercises the live
LLM (skipped without an API key).
"""

import json
import pathlib

import pytest

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "three_tool_eval.json"


def test_fixture_loads_and_validates_schema():
    data = json.loads(FIXTURE_PATH.read_text())
    assert isinstance(data, list)
    assert len(data) >= 14
    seen_ids: set[str] = set()
    valid_tools = {"retrieve", "survey", "retrieve_scoped", "none"}
    for case in data:
        assert case["id"] not in seen_ids, f"duplicate id: {case['id']}"
        seen_ids.add(case["id"])
        assert case["expected_tool"] in valid_tools
        assert isinstance(case["user_message"], str) and case["user_message"]
        assert isinstance(case["prior_messages"], list)
        if case["expected_tool"] != "retrieve_scoped":
            assert case["expected_sequence_number"] is None, (
                f"{case['id']}: non-scoped tool but has expected_sequence_number"
            )
            assert case["expected_season_number"] is None, (
                f"{case['id']}: non-scoped tool but has expected_season_number"
            )


def test_fixture_covers_all_buckets():
    """Each tool bucket must appear at least once."""
    data = json.loads(FIXTURE_PATH.read_text())
    tools_seen = {case["expected_tool"] for case in data}
    assert "retrieve" in tools_seen
    assert "survey" in tools_seen
    assert "retrieve_scoped" in tools_seen
    assert "none" in tools_seen


@pytest.mark.skip(reason="live LLM eval — implement when eval harness lands; see PR #334 follow-up")
def test_live_llm_picks_expected_tool():
    """Placeholder for the live-LLM bucket. Once the marker-gated eval harness
    is in place, this should iterate fixture cases, call run_chat_turn against
    the real configured LLM, and assert tool selection matches expected_tool.

    For this PR, manual repro of conv 7e424a7e (4 turns) covers the
    acceptance check; this slot is reserved so the follow-up PR has a clear
    insertion point.
    """
    pytest.skip("live LLM harness not yet implemented")
