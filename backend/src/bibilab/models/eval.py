from typing import Annotated, Literal

from pydantic import BaseModel, Field


class EvalLLMOverride(BaseModel):
    """Per-call override, merged field-by-field onto backend cfg.ai — an
    omitted field inherits the backend's configured value (see
    eval.py:_merge_ai_config)."""

    # Literal, not free str: every dispatch downstream is `== "anthropic" else
    # <openai branch>`, so a typo would silently pick the wrong wire protocol.
    protocol: Literal["openai", "anthropic"] | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_output_tokens: int | None = None


class EvalChatRequest(BaseModel):
    query: str = Field(..., max_length=10000)
    list_id: str
    llm: EvalLLMOverride | None = None
    language: Literal["zh", "en"] | None = None


class EvalSection(BaseModel):
    index: int
    section_id: str
    source_id: str
    source_title: str
    timestamp_start: float | None
    timestamp_end: float | None
    rerank_score: float | None
    full_text: str
    cited: bool


class EvalFindPassagesCall(BaseModel):
    tool_name: Literal["find_passages"] = "find_passages"
    query: str
    sections: list[EvalSection]
    candidates_evaluated: int | None
    sources_with_hits: int | None
    sources_total: int | None
    reranked: bool | None
    scoped_pool_size: int | None
    facet_scope: dict | None


class EvalReadSectionCall(BaseModel):
    tool_name: Literal["read_section"] = "read_section"
    index: int
    section_id: str
    source_id: str
    source_title: str
    full_text: str
    cited: bool


EvalToolCall = Annotated[EvalFindPassagesCall | EvalReadSectionCall, Field(discriminator="tool_name")]


class EvalChatResponse(BaseModel):
    answer: str
    tool_calls: list[EvalToolCall]
    iterations_used: int
    synthesis_forced: bool
    latency_ms: int
