"""Eval gate for the rewriter prompt. Runs every fixture in
rewriter_eval.json against the live rewriter and asserts that the
extracted intent matches the expected shape.

Skipped unless BIBILAB_RUN_REWRITER_EVAL=1 — the gate hits the LLM and is
not appropriate for every commit. CI enables it on prompt or rewriter
changes only.
"""

import json
import os
from pathlib import Path

import pytest

from bibilab.config import load_config
from bibilab.pipeline.rewriter import PriorUserTurn, run_rewriter

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "rewriter_eval.json").read_text("utf-8"))


@pytest.mark.skipif(
    os.environ.get("BIBILAB_RUN_REWRITER_EVAL") != "1",
    reason="rewriter eval gate; set BIBILAB_RUN_REWRITER_EVAL=1 to run",
)
@pytest.mark.parametrize("case", FIXTURES, ids=[c["id"] for c in FIXTURES])
def test_rewriter_eval_case(case):
    cfg = load_config().ai
    prior = [PriorUserTurn(text=p["text"], retrieved=p["retrieved"]) for p in case["prior"]]
    intent, telemetry = run_rewriter(current=case["current"], prior=prior, cfg=cfg)
    expect = case["expect"]

    assert intent.retrieve == expect["retrieve"], f"case {case['id']}: retrieve mismatch (got telemetry={telemetry})"
    if "mode" in expect:
        assert intent.mode == expect["mode"]
    if "sequence_number" in expect:
        assert intent.sequence_number == expect["sequence_number"]
    if "query_inherits_from_prior_index" in expect:
        expected_query = case["prior"][expect["query_inherits_from_prior_index"]]["text"]
        assert intent.query == expected_query
