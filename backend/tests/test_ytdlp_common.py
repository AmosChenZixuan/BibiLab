"""Tests for the shared yt-dlp subprocess runner (_run_subprocess/run_ytdlp)
and the small helpers around it (parse_download_path). The aria2c_argv and
raise_mapped branches are covered by the adapter-level download tests, where
they're exercised end-to-end with the real argv and real error patterns."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

from bibilab.adapters import _ytdlp_common
from bibilab.adapters.base import DownloadError


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
    """A child that traps SIGTERM must still die: killpg-SIGTERM alone won't
    stop it, so the runner has to escalate to killpg-SIGKILL after the grace
    period. monkeypatch the timeout so the test runs in well under a second."""
    monkeypatch.setattr(_ytdlp_common, "TERMINATE_TIMEOUT", 0.3)

    captured: dict = {}
    orig_create = asyncio.create_subprocess_exec

    async def spying_create(*args, **kwargs):
        proc = await orig_create(*args, **kwargs)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spying_create)

    script = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    task = asyncio.create_task(_ytdlp_common._run_subprocess([sys.executable, "-c", script]))
    await asyncio.sleep(0.2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert captured["proc"].returncode is not None


@pytest.mark.asyncio
async def test_cancel_swallows_process_lookup_error_on_already_exited_child(monkeypatch):
    """killpg on a process that already exited raises ProcessLookupError;
    that must be swallowed, not surface in place of the CancelledError."""

    class FakeProc:
        pid = 12345
        returncode = 0

        async def communicate(self):
            raise asyncio.CancelledError()

        async def wait(self):
            return 0

    captured: dict = {}

    async def fake_create(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    def fake_killpg(pid, sig):
        captured["sig"] = sig
        raise ProcessLookupError()

    monkeypatch.setattr(_ytdlp_common.os, "killpg", fake_killpg)

    with pytest.raises(asyncio.CancelledError):
        await _ytdlp_common._run_subprocess(["irrelevant"])

    # killpg was attempted with SIGTERM (the first escalation step);
    # the SIGKILL escalation is skipped because wait() already returned.
    assert captured["sig"] == _ytdlp_common.signal.SIGTERM


@pytest.mark.asyncio
async def test_run_ytdlp_invokes_module_never_bare_binary(monkeypatch):
    """yt-dlp must run as `sys.executable -m yt_dlp`, never a bare `yt-dlp`
    binary that may not be on PATH in a container or uv-managed venv."""
    captured = {}

    async def fake_run_subprocess(argv):
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
