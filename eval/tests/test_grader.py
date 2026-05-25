from eval.grader import (
    build_context_relevance_prompt,
    build_groundedness_prompt,
    build_answer_relevance_prompt,
    parse_grade_response,
)


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
