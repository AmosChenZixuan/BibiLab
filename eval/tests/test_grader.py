import asyncio

from eval.grader import (
    build_abstention_prompt,
    build_context_relevance_prompt,
    build_coverage_groundedness_prompt,
    build_groundedness_prompt,
    build_answer_relevance_prompt,
    parse_grade_response,
    _grade_case,
)
from eval.models import RunCaseResult


def test_build_context_relevance_prompt():
    prompt = build_context_relevance_prompt(
        question="什么是RAG？",
        chunks_text="[1]: RAG stands for Retrieval-Augmented Generation...",
    )
    assert "什么是RAG" in prompt
    assert "RAG stands for" in prompt
    assert "1-5" in prompt


def test_build_groundedness_prompt():
    prompt = build_groundedness_prompt(
        answer="RAG is a technique...",
        chunks_text="[1]: RAG stands for...",
    )
    assert "RAG is a technique" in prompt
    assert "RAG stands for" in prompt


def test_build_answer_relevance_prompt():
    prompt = build_answer_relevance_prompt(
        question="什么是RAG？",
        answer="RAG is a technique...",
    )
    assert "什么是RAG" in prompt
    assert "RAG is a technique" in prompt


def test_parse_grade_response_valid():
    response = '{"score": 4, "reasoning": "Good coverage but missed one aspect."}'
    score, reasoning = parse_grade_response(response)
    assert score == 4
    assert "Good coverage" in reasoning


def test_parse_grade_response_invalid_json():
    score, reasoning = parse_grade_response("not json")
    assert score is None
    assert "Failed to parse" in reasoning


def test_parse_grade_response_out_of_range():
    score, reasoning = parse_grade_response('{"score": 7, "reasoning": "great"}')
    assert score is None


def test_parse_grade_response_with_markdown_fences():
    response = '```json\n{"score": 3, "reasoning": "mixed"}\n```'
    score, reasoning = parse_grade_response(response)
    assert score == 3
    assert reasoning == "mixed"


def test_build_abstention_prompt_includes_expected_and_answer():
    prompt = build_abstention_prompt(
        question="为什么主角要做出那个决定？",
        expected_answer_draft="资料里提到了这件事，但没有解释原因。",
        answer="因为他想要复仇。",
    )
    assert "为什么主角要做出那个决定" in prompt
    assert "资料里提到了这件事" in prompt
    assert "因为他想要复仇" in prompt
    # the rubric must frame absent context as the EXPECTED outcome, not a failure
    assert "EXPECTED" in prompt or "not a failure" in prompt


def test_build_coverage_groundedness_prompt_accepts_summaries():
    prompt = build_coverage_groundedness_prompt(
        answer="这一集主要讲了主角的身世。",
        chunks_text="[1] 主角身世的概要……",
    )
    assert "summary-derived" in prompt or "summaries" in prompt
    assert "do NOT penalize" in prompt


def test_grade_case_causal_absent_correct_abstention_passes(monkeypatch):
    # A correct abstention has NO retrieved context (the corpus lacks the cause).
    # The category-agnostic CR/G rubric would score that 1; the abstention branch
    # must instead score the correct "not covered" answer as a PASS across all dims.
    monkeypatch.setattr(
        "eval.grader._call_llm",
        lambda p, *a, **k: '{"score": 5, "reasoning": "correctly abstained"}',
    )
    case = RunCaseResult(case_id="c1", answer="资料里没有解释原因。", llm_context=[])
    grade = asyncio.run(
        _grade_case(case, "为什么X会发生？", "causal_absent", "资料里没有解释其原因。", ai_cfg=None)
    )
    assert grade.context_relevance == 5
    assert grade.groundedness == 5
    assert grade.answer_relevance == 5
