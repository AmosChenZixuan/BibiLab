from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

CATEGORY = Literal[
    "single_fact",
    "locate",
    "enumeration",
    "comparison",
    "multi_hop",
    "coverage",
    "causal_absent",
    "temporal",
    "entity_profile",
]

# Single source of truth for the taxonomy — CLI defaults derive from the Literal.
ALL_CATEGORIES: tuple[str, ...] = get_args(CATEGORY)
DEFAULT_CATEGORIES = ",".join(ALL_CATEGORIES)
DEFAULT_FLOOR = 3


class ProfileSnapshot(BaseModel):
    """Frozen record of which LLM was used. Stored on EvalRun / GradedRun."""
    protocol: str = "openai"
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None


class Evidence(BaseModel):
    source_id: str
    section_seq: int
    snippet: str = ""  # short quoted span the claim came from (for human review)


class EvalCase(BaseModel):
    id: str
    category: CATEGORY
    question: str
    expected_answer_draft: str = ""
    locked: bool = False
    notes: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class EvalSet(BaseModel):
    id: str
    list_id: str
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
    # Exact LLM-visible tool context per call, in message order — what the
    # model actually read. Grading judges against this for parity.
    llm_context: list[str] = Field(default_factory=list)
    llm_duration_ms: int = 0
    error: str | None = None


class EvalRun(BaseModel):
    id: str
    eval_set_id: str
    test_profile: ProfileSnapshot = Field(default_factory=ProfileSnapshot)
    timestamp: str
    cases: list[RunCaseResult] = Field(default_factory=list)


class GradeResult(BaseModel):
    case_id: str
    # 1-5 = judge scores; None = grading failed (see grader._grade_one).
    context_relevance: int | None = Field(default=None, ge=1, le=5)
    context_relevance_reasoning: str = ""
    groundedness: int | None = Field(default=None, ge=1, le=5)
    groundedness_reasoning: str = ""
    answer_relevance: int | None = Field(default=None, ge=1, le=5)
    answer_relevance_reasoning: str = ""
    llm_duration_ms: int = 0


class GradedRun(BaseModel):
    run_id: str
    grade_profile: ProfileSnapshot = Field(default_factory=ProfileSnapshot)
    timestamp: str
    grades: list[GradeResult] = Field(default_factory=list)
