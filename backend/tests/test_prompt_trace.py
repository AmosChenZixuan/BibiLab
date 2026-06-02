"""Tests for opt-in per-message LLM prompt-trace dump (#393)."""

import json
import logging
from pathlib import Path

from bibilab.config import RagConfig


def test_rag_config_default_debug_prompts_is_false():
    """Off by default — opt-in flag, zero behavior change for existing users."""
    cfg = RagConfig()
    assert cfg.debug_prompts is False


def test_dump_prompt_trace_writes_json(monkeypatch, tmp_path: Path):
    """Helper writes a valid JSON file with system + iterations."""
    from bibilab.routers.chat import _dump_prompt_trace

    debug_dir = tmp_path / "debug"
    iterations = [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "find_passages", "description": "x", "parameters": {}}],
        }
    ]
    _dump_prompt_trace("m1", "sys prompt", iterations, debug_dir)

    out = debug_dir / "m1.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data == {
        "system": "sys prompt",
        "iterations": iterations,
    }


def test_dump_prompt_trace_skips_empty_iterations(tmp_path: Path):
    """If there are no iterations, do not write a file (nothing to debug)."""
    from bibilab.routers.chat import _dump_prompt_trace

    debug_dir = tmp_path / "debug"
    _dump_prompt_trace("m1", "sys", [], debug_dir)

    assert not (debug_dir / "m1.json").exists()
    assert not debug_dir.exists()


def test_dump_prompt_trace_allows_null_system(tmp_path: Path):
    """system may be None (e.g. whitespace-only after build_grounding_prompt) — store as null."""
    from bibilab.routers.chat import _dump_prompt_trace

    debug_dir = tmp_path / "debug"
    _dump_prompt_trace("m1", None, [{"messages": [], "tools": []}], debug_dir)

    data = json.loads((debug_dir / "m1.json").read_text())
    assert data["system"] is None


def test_dump_prompt_trace_swallows_write_errors(tmp_path: Path, caplog):
    """A write failure (e.g. permission denied) must not propagate — it must log."""
    from bibilab.routers import chat as chat_module

    # Use a path where mkdir will fail: a file exists at the location.
    bad = tmp_path / "blocking_file"
    bad.write_text("not a dir")
    # _dump_prompt_trace calls debug_dir.mkdir(parents=True, exist_ok=True)
    # which will raise NotADirectoryError on Linux when parent is a file.

    with caplog.at_level(logging.WARNING, logger="bibilab.routers.chat"):
        chat_module._dump_prompt_trace(
            "m1",
            "sys",
            [{"messages": [], "tools": []}],
            bad,
        )
    # No exception propagated. The blocking file is untouched.
    assert bad.read_text() == "not a dir"
    assert any("dump_prompt_trace_failed" in r.message for r in caplog.records)
