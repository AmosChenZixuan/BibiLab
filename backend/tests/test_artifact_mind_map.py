"""Mind-map artifact (type='mind_map').

The mind-map path forks inside _run_artifact_job: meta.prompt is replaced
by _MIND_MAP_PROMPT, the result is fed to the standard _refine_artifact
section-batched pipeline, and the LLM output is contract-checked for
exactly one ```mermaid fence before the file is written.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from bibilab.config import BibilabConfig
from bibilab.db import bootstrap_db, create_list, get_artifact, get_job
from bibilab.pipeline.audio import PipelineError
from bibilab.pipeline.section import Section
from bibilab.pipeline.transcribe import WhisperSegment
from bibilab.worker import (
    _MIND_MAP_PROMPT,
    WorkerLoop,
    _refine_mind_map,
    _validate_mind_map_fence,
)
from tests.factories import SourceFactory

pytestmark = pytest.mark.integration

# MindMapResult import will be enabled as each slice lands.
# from bibilab.worker import MindMapResult


# --- MindMapResult model --------------------------------------------------


def test_mind_map_result_accepts_valid_input():
    """MindMapResult(name, root) constructs from a plain dict root and
    exposes both fields. This is the seam that lets the LLM emit a single
    JSON object with no envelope wrapper."""
    from bibilab.worker import MindMapResult

    mm = MindMapResult(name="Topic", root={"label": "Root", "children": []})
    assert mm.name == "Topic"
    assert mm.root == {"label": "Root", "children": []}


def test_mind_map_result_rejects_missing_root():
    """Missing `root` is a contract violation; Pydantic must raise."""
    from pydantic import ValidationError

    from bibilab.worker import MindMapResult

    with pytest.raises(ValidationError):
        MindMapResult(name="Topic")  # type: ignore[call-arg]


def test_mind_map_result_rejects_non_dict_root():
    """`root` must be a dict (recursive tree); a string is rejected."""
    from pydantic import ValidationError

    from bibilab.worker import MindMapResult

    with pytest.raises(ValidationError):
        MindMapResult(name="Topic", root="not a dict")  # type: ignore[arg-type]


# --- _render_mind_map_markdown --------------------------------------------


def test_render_mind_map_markdown_roundtrips_through_fence_validator():
    """`_render_mind_map_markdown(mm)` produces a content string whose
    single ```json fence re-parses to `{"root": mm.root}` via
    `_validate_mind_map_fence`. This is the write-side symmetry check:
    whatever the LLM returns, the on-disk file round-trips back to the
    same tree."""
    from bibilab.worker import MindMapResult, _render_mind_map_markdown

    root = {"label": "Topic", "children": [{"label": "Branch"}]}
    mm = MindMapResult(name="Topic Map", root=root)
    content = _render_mind_map_markdown(mm)
    assert _validate_mind_map_fence(content) == {"root": root}


# --- _validate_mind_map_fence ---------------------------------------------


def test_validate_mind_map_fence_single_returns_source():
    """Well-formed content with one closed JSON fence → returns the inner
    JSON, stripped of leading/trailing newlines. The four contract
    failures (0/2+/unclosed/multiple) and the happy path collapse into
    one test because the function is a pure regex split — one test per
    branch is overkill."""
    body = '# Topic\n\n```json\n{\n  "name": "x",\n  "root": {"label": "T"}\n}\n```\n'
    assert _validate_mind_map_fence(body) == {"name": "x", "root": {"label": "T"}}

    # Zero fences → PipelineError.
    with pytest.raises(PipelineError, match="exactly one"):
        _validate_mind_map_fence("no fence here")
    # Two fences → PipelineError.
    with pytest.raises(PipelineError, match="exactly one"):
        _validate_mind_map_fence("```json\na\n```\n```json\nb\n```")
    # Unclosed fence → PipelineError.
    with pytest.raises(PipelineError, match="not closed"):
        _validate_mind_map_fence("```json\nstuff\n")


def test_mind_map_prompt_instructs_only_name_and_root():
    """The prompt must ask the LLM for a single JSON object with `name`
    and `root`. Any directive that points the model at a `content`
    envelope or a fenced JSON block re-introduces the nesting pattern
    the refactor kills. These assertions lock the structural intent."""
    assert '"name"' in _MIND_MAP_PROMPT
    assert '"root"' in _MIND_MAP_PROMPT
    # No envelope wrapper directive — `content` is not a field the LLM
    # should emit.
    assert '"content"' not in _MIND_MAP_PROMPT
    # No fence directive — the worker constructs the file body itself;
    # asking the LLM to wrap a JSON block in ```json re-opens the
    # nesting class this PR eliminates.
    assert "```json" not in _MIND_MAP_PROMPT


# --- _run_artifact_job dispatch + happy path ------------------------------


@pytest.mark.asyncio
async def test_run_artifact_job_mind_map_dispatches_and_validates(tmp_bibilab_home):
    """type='mind_map' → prompt is rebound to _MIND_MAP_PROMPT,
    `_refine_mind_map` produces a MindMapResult, the worker renders the
    markdown file body via `_render_mind_map_markdown`, and the artifact
    row + content file land. End-to-end: this is the single test that
    proves the new path works."""
    from bibilab.worker import MindMapResult

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build(
        "list-1",
        video_id="BV1",
        segments=[
            WhisperSegment(start=0.0, end=1.0, text="hello", speaker="SPK_0"),
            WhisperSegment(start=1.0, end=2.0, text="world", speaker="SPK_0"),
        ],
        sections=[
            Section(seg_start=0, seg_end=1, token_count=2, timestamp_start=0.0, timestamp_end=2.0),
        ],
    )

    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)
    job_id = "job-mm-1"
    job = {
        "id": job_id,
        "meta": json.dumps(
            {
                "list_id": "list-1",
                "artifact_id": "art-mm-1",
                "type": "mind_map",
                "prompt": "(user typed something the worker will ignore)",
                "source_ids": [source_id],
            }
        ),
    }

    async def _fake_refine_mind_map(*, sections, cfg, ui_lang=None):
        return MindMapResult(
            name="Topic Map",
            root={
                "label": "Topic",
                "children": [{"label": "Branch", "children": [{"label": "Detail"}]}],
            },
        )

    with patch("bibilab.worker._refine_mind_map", side_effect=_fake_refine_mind_map) as mock_refine:
        await worker._run_artifact_job(job)

    assert mock_refine.call_count == 1
    art = await get_artifact("art-mm-1")
    assert art["status"] == "completed"
    assert art["type"] == "mind_map"
    # The view-prompt modal shows the directive the LLM was actually given,
    # not the user-typed text.
    assert art["prompt"] == _MIND_MAP_PROMPT
    content_path = tmp_bibilab_home / "artifacts" / "list-1" / "art-mm-1.md"
    assert content_path.exists()
    # Worker-rendered file body: round-trips through the fence validator
    # to the same tree the LLM returned (Slice B invariant).
    parsed = _validate_mind_map_fence(content_path.read_text())
    assert parsed == {
        "root": {
            "label": "Topic",
            "children": [{"label": "Branch", "children": [{"label": "Detail"}]}],
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_error",
    [
        # PipelineError message from malformed JSON (LLM retry exhausts).
        "LLM mind_map batch 1/1",
        # PipelineError message from missing `root` (Pydantic rejects).
        "LLM mind_map batch 1/1",
    ],
)
async def test_run_artifact_job_mind_map_bad_output_fails_job(tmp_bibilab_home, expected_error):
    """When `_refine_mind_map` raises (LLM returned bad JSON or Pydantic
    rejected a missing `root`), the worker propagates the failure as a
    job-level error: status=failed, no artifact row, no file written.
    The failure path used to be the fence validator (which is now
    unreachable from the dispatch — content is rendered from a parsed
    MindMapResult, not validated from raw LLM output)."""
    from bibilab.db import create_job

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build(
        "list-1",
        video_id="BV1",
        segments=[WhisperSegment(start=0.0, end=1.0, text="hi", speaker="SPK_0")],
        sections=[Section(seg_start=0, seg_end=0, token_count=1, timestamp_start=0.0, timestamp_end=1.0)],
    )

    actual_id = await create_job(
        "artifact",
        {
            "list_id": "list-1",
            "artifact_id": "art-bad",
            "type": "mind_map",
            "prompt": "ignored",
            "source_ids": [source_id],
        },
    )
    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)
    job = {
        "id": actual_id,
        "meta": json.dumps(
            {
                "list_id": "list-1",
                "artifact_id": "art-bad",
                "type": "mind_map",
                "prompt": "ignored",
                "source_ids": [source_id],
            }
        ),
    }

    async def _fake_refine_mind_map(*, sections, cfg, ui_lang=None):
        raise PipelineError("LLM mind_map batch 1/1 exhausted all retries: simulated")

    with patch("bibilab.worker._refine_mind_map", side_effect=_fake_refine_mind_map):
        await worker._run_artifact_job(job)

    job_row = await get_job(actual_id)
    assert job_row["status"] == "failed"
    assert expected_error in (job_row["error"] or "")
    assert await get_artifact("art-bad") is None
    assert not (tmp_bibilab_home / "artifacts" / "list-1" / "art-bad.md").exists()


# --- _run_artifact_job dispatch (mind_map → _refine_mind_map) -------------


@pytest.mark.asyncio
async def test_run_artifact_job_mind_map_uses_refine_mind_map(tmp_bibilab_home):
    """type='mind_map' jobs route through `_refine_mind_map` (NOT
    `_refine_artifact`). The on-disk file content is rendered by
    `_render_mind_map_markdown` from the parsed MindMapResult. Frontend
    reads `parsed.root` from the fence — unchanged contract."""
    from bibilab.worker import MindMapResult

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build(
        "list-1",
        video_id="BV1",
        segments=[
            WhisperSegment(start=0.0, end=1.0, text="hello", speaker="SPK_0"),
            WhisperSegment(start=0.0, end=1.0, text="world", speaker="SPK_0"),
        ],
        sections=[Section(seg_start=0, seg_end=1, token_count=2, timestamp_start=0.0, timestamp_end=1.0)],
    )
    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)
    job = {
        "id": "job-mm-new",
        "meta": json.dumps(
            {
                "list_id": "list-1",
                "artifact_id": "art-mm-new",
                "type": "mind_map",
                "prompt": "(user typed something the worker will ignore)",
                "source_ids": [source_id],
            }
        ),
    }

    async def _fake_refine_mind_map(*, sections, cfg, ui_lang=None):
        return MindMapResult(
            name="Topic Map",
            root={"label": "Topic", "children": [{"label": "Branch"}]},
        )

    with (
        patch("bibilab.worker._refine_mind_map", side_effect=_fake_refine_mind_map) as mock_mm,
        patch("bibilab.worker._refine_artifact") as mock_ar,
    ):
        await worker._run_artifact_job(job)

    # mind_map dispatched through _refine_mind_map, never _refine_artifact.
    assert mock_mm.call_count == 1
    assert mock_ar.call_count == 0
    art = await get_artifact("art-mm-new")
    assert art["status"] == "completed"
    assert art["name"] == "Topic Map"
    assert art["prompt"] == _MIND_MAP_PROMPT
    content_path = tmp_bibilab_home / "artifacts" / "list-1" / "art-mm-new.md"
    assert content_path.exists()
    # Worker-rendered content, not LLM-emitted.
    parsed = _validate_mind_map_fence(content_path.read_text())
    assert parsed == {"root": {"label": "Topic", "children": [{"label": "Branch"}]}}


# --- _refine_mind_map -----------------------------------------------------


def _section_view(source_id: str, title: str, seq: int, tokens: int) -> object:
    """Tiny _SectionView stand-in for refine tests; the same shape the
    worker passes around (text + token_count + headers)."""
    from bibilab.worker import _SectionView

    return _SectionView(
        source_id=source_id,
        source_title=title,
        seq=seq,
        timestamp_start=0.0,
        timestamp_end=1.0,
        text=f"{title} text.",
        token_count=tokens,
    )


@pytest.mark.asyncio
async def test_refine_mind_map_single_batch_parses_mind_map_result(mock_call_llm):
    """Sections fit in one batch → one LLM call. The LLM returns a JSON
    object with `name` + `root` only (no `content` envelope, no fence);
    `_refine_mind_map` parses it as MindMapResult and returns it."""
    cfg = BibilabConfig()
    cfg.ai.context_window = 32_000
    cfg.ai.max_output_tokens = 4_000
    sections = [
        _section_view("src-A", "A", 0, 2),
        _section_view("src-B", "B", 0, 2),
    ]
    mock_call_llm.return_value = json.dumps({"name": "Topic Map", "root": {"label": "Topic", "children": []}})

    result = await _refine_mind_map(sections=sections, cfg=cfg, ui_lang="en")

    from bibilab.worker import MindMapResult

    assert isinstance(result, MindMapResult)
    assert result.name == "Topic Map"
    assert result.root == {"label": "Topic", "children": []}
    assert mock_call_llm.call_count == 1


@pytest.mark.asyncio
async def test_refine_mind_map_multi_batch_refines_running_draft(mock_call_llm):
    """Sections don't fit in one batch → 2 LLM calls. Batch 1 produces
    the initial MindMapResult; batch 2 refines it with the running draft
    shown in the prompt."""
    cfg = BibilabConfig()
    # Tiny budget: each section alone exceeds → 2 batches.
    cfg.ai.context_window = 60 + 2 * 4000 + 500
    cfg.ai.max_output_tokens = 4000
    sections = [
        _section_view("src-1", "A", 0, 50),
        _section_view("src-1", "A", 1, 50),
    ]
    mock_call_llm.side_effect = [
        json.dumps({"name": "Initial", "root": {"label": "A"}}),
        json.dumps({"name": "Refined", "root": {"label": "A", "children": [{"label": "B"}]}}),
    ]

    result = await _refine_mind_map(sections=sections, cfg=cfg, ui_lang="en")

    assert mock_call_llm.call_count == 2
    # Final result is the second call's parsed output.
    assert result.name == "Refined"
    assert result.root["label"] == "A"
    assert result.root["children"] == [{"label": "B"}]
    # The second prompt must contain the running draft (first call's
    # `name` and `root`) so the LLM can integrate.
    second_prompt = mock_call_llm.call_args_list[1][0][0]
    assert "Initial" in second_prompt
    assert '"label": "A"' in second_prompt


@pytest.mark.asyncio
async def test_refine_mind_map_malformed_json_raises_pipeline_error(mock_call_llm):
    """LLM returns invalid JSON → retry ladder exhausts → PipelineError."""
    cfg = BibilabConfig()
    cfg.ai.context_window = 32_000
    cfg.ai.max_output_tokens = 4_000
    sections = [_section_view("src-A", "A", 0, 2)]
    mock_call_llm.return_value = "{not valid json"

    with pytest.raises(PipelineError):
        await _refine_mind_map(sections=sections, cfg=cfg, ui_lang="en")


@pytest.mark.asyncio
async def test_refine_mind_map_missing_root_raises_pipeline_error(mock_call_llm):
    """LLM returns valid JSON but missing the `root` key → Pydantic
    ValidationError → retry ladder exhausts → PipelineError. This is the
    failure mode the original `name` collision was producing."""
    cfg = BibilabConfig()
    cfg.ai.context_window = 32_000
    cfg.ai.max_output_tokens = 4_000
    sections = [_section_view("src-A", "A", 0, 2)]
    mock_call_llm.return_value = json.dumps({"name": "X"})  # no `root`

    with pytest.raises(PipelineError):
        await _refine_mind_map(sections=sections, cfg=cfg, ui_lang="en")
