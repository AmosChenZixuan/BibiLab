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
    """type='mind_map' → prompt is replaced with _MIND_MAP_PROMPT, the
    standard _refine_artifact runs, the fence validator passes, and the
    artifact row + content file land. End-to-end: this is the single test
    that proves the new path works."""
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
    llm_content = (
        "```json\n"
        '{\n  "name": "Topic Map",\n  "root": {"label": "Topic", "children": [\n'
        '    {"label": "Branch", "children": [{"label": "Detail"}]}\n'
        "  ]}\n"
        "}\n"
        "```\n"
    )

    async def _fake_refine(*, prompt, sections, cfg, ui_lang=None):
        # The worker must have rebound the prompt to _MIND_MAP_PROMPT before
        # this call — that's the whole point of the fork.
        assert prompt == _MIND_MAP_PROMPT
        from bibilab.worker import ArtifactResult

        return ArtifactResult(name="Topic Map", content=llm_content)

    with patch("bibilab.worker._refine_artifact", side_effect=_fake_refine) as mock_refine:
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
    assert content_path.read_text() == llm_content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_content,expected_error",
    [
        # Two fences — fence validator raises "exactly one".
        ("```json\na\n```\n```json\nb\n```", "exactly one"),
        # One fence, valid syntax, but no `root` key — shape check raises.
        ('```json\n{"name": "x"}\n```\n', "`root`"),
    ],
)
async def test_run_artifact_job_mind_map_bad_output_fails_job(tmp_bibilab_home, bad_content, expected_error):
    """LLM output that fails fence or shape validation marks the job failed,
    writes no file, and creates no artifact row."""
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

    async def _fake_refine(*, prompt, sections, cfg, ui_lang=None):
        from bibilab.worker import ArtifactResult

        return ArtifactResult(name="Bad", content=bad_content)

    with patch("bibilab.worker._refine_artifact", side_effect=_fake_refine):
        await worker._run_artifact_job(job)

    job_row = await get_job(actual_id)
    assert job_row["status"] == "failed"
    assert expected_error in (job_row["error"] or "")
    assert await get_artifact("art-bad") is None
    assert not (tmp_bibilab_home / "artifacts" / "list-1" / "art-bad.md").exists()
