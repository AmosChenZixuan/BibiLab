"""Tests for the shared yt-dlp subprocess runner (_run_subprocess/run_ytdlp)
and the small helpers around it (aria2c_argv, parse_download_path, raise_mapped)."""

import asyncio
import re
import sys
import time
from pathlib import Path

import pytest

from bibilab.adapters import _ytdlp_common
from bibilab.adapters.base import AuthRequiredError, DownloadError


@pytest.mark.asyncio
async def test_cancel_terminates_long_lived_child_within_one_second(monkeypatch):
    """Cancelling the coroutine mid-download terminates the child instead of
    letting it run to completion, and does so fast — this is the whole point
    of moving off asyncio.to_thread, which cannot interrupt a running thread."""
    captured: dict = {}
    orig_create = asyncio.create_subprocess_exec

    async def spying_create(*args, **kwargs):
        proc = await orig_create(*args, **kwargs)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spying_create)

    task = asyncio.create_task(_ytdlp_common._run_subprocess([sys.executable, "-c", "import time; time.sleep(60)"]))
    await asyncio.sleep(0.2)  # let the child actually start before cancelling

    start = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert captured["proc"].returncode is not None


@pytest.mark.asyncio
async def test_cancel_escalates_to_sigkill_when_child_ignores_sigterm(monkeypatch):
    """A child that traps SIGTERM must still die: terminate() alone won't
    stop it, so the runner has to escalate to kill() after the grace period."""
    captured: dict = {}
    orig_create = asyncio.create_subprocess_exec

    async def spying_create(*args, **kwargs):
        proc = await orig_create(*args, **kwargs)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spying_create)

    script = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    task = asyncio.create_task(_ytdlp_common._run_subprocess([sys.executable, "-c", script], terminate_timeout=0.3))
    await asyncio.sleep(0.2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert captured["proc"].returncode is not None


@pytest.mark.asyncio
async def test_cancel_swallows_process_lookup_error_on_already_exited_child(monkeypatch):
    """terminate() on a process that already exited raises ProcessLookupError;
    that must be swallowed, not surface in place of the CancelledError."""

    class FakeProc:
        returncode = 0

        async def communicate(self):
            raise asyncio.CancelledError()

        def terminate(self):
            raise ProcessLookupError()

        def kill(self):
            raise AssertionError("kill() should not run — wait() already returned")

        async def wait(self):
            return 0

    async def fake_create(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    with pytest.raises(asyncio.CancelledError):
        await _ytdlp_common._run_subprocess(["irrelevant"])


@pytest.mark.asyncio
async def test_run_ytdlp_invokes_module_never_bare_binary(monkeypatch):
    """yt-dlp must run as `sys.executable -m yt_dlp`, never a bare `yt-dlp`
    binary that may not be on PATH in a container or uv-managed venv."""
    captured = {}

    async def fake_run_subprocess(argv, *, terminate_timeout=_ytdlp_common.TERMINATE_TIMEOUT):
        captured["argv"] = argv
        return "out", "err", 0

    monkeypatch.setattr(_ytdlp_common, "_run_subprocess", fake_run_subprocess)

    result = await _ytdlp_common.run_ytdlp(["-f", "best", "url"])

    assert captured["argv"] == [sys.executable, "-m", "yt_dlp", "-f", "best", "url"]
    assert result == ("out", "err", 0)


def test_parse_download_path_returns_last_nonempty_line():
    stdout = "[download] some progress line\n\n/home/user/.bibilab/downloads/BV1.mp4\n"
    assert _ytdlp_common.parse_download_path(stdout) == Path("/home/user/.bibilab/downloads/BV1.mp4")


def test_parse_download_path_raises_when_no_path_printed():
    with pytest.raises(DownloadError):
        _ytdlp_common.parse_download_path("   \n\n")


def test_aria2c_argv_present_when_available(monkeypatch):
    monkeypatch.setattr(_ytdlp_common.shutil, "which", lambda name: "/usr/bin/aria2c")
    assert _ytdlp_common.aria2c_argv(16) == [
        "--downloader",
        "aria2c",
        "--downloader-args",
        "aria2c:-x16 -s16 -k1M --file-allocation=none",
    ]


def test_aria2c_argv_empty_when_absent(monkeypatch):
    monkeypatch.setattr(_ytdlp_common.shutil, "which", lambda name: None)
    assert _ytdlp_common.aria2c_argv(16) == []


def test_raise_mapped_auth_family_raises_auth_required():
    with pytest.raises(AuthRequiredError):
        _ytdlp_common.raise_mapped("Sign in to confirm you're not a bot", re.compile("sign in", re.IGNORECASE))


def test_raise_mapped_override_pattern_uses_fixed_message():
    with pytest.raises(DownloadError) as exc_info:
        _ytdlp_common.raise_mapped(
            "no video formats found",
            re.compile("nope", re.IGNORECASE),
            message_overrides=((re.compile("no video formats"), "image post"),),
        )
    assert exc_info.value.message == "image post"


def test_raise_mapped_default_strips_ansi_and_appends_hint():
    with pytest.raises(DownloadError) as exc_info:
        _ytdlp_common.raise_mapped("\x1b[31mboom\x1b[0m", re.compile("nope", re.IGNORECASE), hint=" hint text")
    assert exc_info.value.message == "boom hint text"


def test_raise_mapped_preserves_cause_chain():
    cause = ValueError("orig")
    with pytest.raises(DownloadError) as exc_info:
        _ytdlp_common.raise_mapped("boom", re.compile("nope", re.IGNORECASE), cause=cause)
    assert exc_info.value.__cause__ is cause
