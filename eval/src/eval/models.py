from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CATEGORY = Literal["narrow", "broad", "cross_ref", "ambiguous", "absence", "temporal"]


class EvalCase(BaseModel):
    id: str
    category: CATEGORY
    question: str
    expected_answer_draft: str = ""
    expected_sources: list[str] = Field(default_factory=list)
    locked: bool = False
    notes: str = ""


class EvalSet(BaseModel):
    id: str
    list_id: str
    generate_profile: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    cases: list[EvalCase] = Field(default_factory=list)

    @property
    def locked_cases(self) -> list[EvalCase]:
        return [c for c in self.cases if c.locked]


class RunCaseResult(BaseModel):
    case_id: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    rag_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_blocks: list[dict[str, Any]] = Field(default_factory=list)
    llm_duration_ms: int = 0
    error: str | None = None


class EvalRun(BaseModel):
    id: str
    eval_set_id: str
    test_profile: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    cases: list[RunCaseResult] = Field(default_factory=list)


class GradeResult(BaseModel):
    case_id: str
    context_relevance: int = Field(ge=0, le=5)
    context_relevance_reasoning: str = ""
    groundedness: int = Field(ge=0, le=5)
    groundedness_reasoning: str = ""
    answer_relevance: int = Field(ge=0, le=5)
    answer_relevance_reasoning: str = ""
    llm_duration_ms: int = 0


class GradedRun(BaseModel):
    run_id: str
    grade_profile: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    grades: list[GradeResult] = Field(default_factory=list)
