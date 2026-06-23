"""End-to-end check that a server returning fewer bytes than Content-Length
surfaces as DownloadError — no silent short file.

Spins up a tiny aiohttp server in-process and runs the real pypdl against it.
"""

import asyncio
import threading

import pytest

pytestmark = pytest.mark.integration


class _TruncatingHandler:
    def __init__(self, declared_length: int, actual_length: int) -> None:
        self.declared_length = declared_length
        self.actual_length = actual_length

    async def handle(self, request):
        from aiohttp import web

        return web.Response(
            body=b"x" * self.actual_length,
            headers={"Content-Length": str(self.declared_length)},
        )


def _start_server(handler: _TruncatingHandler) -> tuple[str, callable]:
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/file.bin", handler.handle)
    runner = web.AppRunner(app)

    async def _run():
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner.addresses[0][1]

    loop = asyncio.new_event_loop()
    port_holder: list[int] = []

    def _runner():
        asyncio.set_event_loop(loop)
        port_holder.append(loop.run_until_complete(_run()))
        loop.run_forever()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    for _ in range(50):
        if port_holder:
            break
        threading.Event().wait(0.05)

    def stop():
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)

    return f"http://127.0.0.1:{port_holder[0]}/file.bin", stop


class _FakeYDL:
    def __init__(self, info: dict) -> None:
        self._info = info

    def __call__(self, opts):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=False):
        return self._info


@pytest.fixture
def adapter_against_local_server(tmp_path, monkeypatch):
    """Yield (adapter, url) where yt-dlp is patched to point at a local server
    serving (declared, actual) byte counts. Caller calls adapter.download(...) and
    inspects the result."""

    servers: list = []

    def _setup(video_id: str, declared: int, actual: int) -> tuple:
        handler = _TruncatingHandler(declared, actual)
        url, stop = _start_server(handler)
        servers.append(stop)

        from bibilab.adapters.bilibili import BilibiliAdapter
        from bibilab.config import load_config as _lc
        _lc()

        info = {
            "url": url,
            "ext": "m4a",
            "protocol": "https",
            "fragments": None,
            "http_headers": {},
            "filesize": declared,
        }
        monkeypatch.setattr("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", _FakeYDL(info))
        monkeypatch.setattr("bibilab.adapters.bilibili.bibilab_home", lambda: tmp_path)

        return BilibiliAdapter(cookie=""), url

    yield _setup

    for stop in servers:
        stop()


def test_truncated_response_raises_download_error(adapter_against_local_server) -> None:
    """Declared Content-Length=1000, server returns 500 → DownloadError."""
    from bibilab.adapters.base import DownloadError

    adapter, _ = adapter_against_local_server("BVtrunc", declared=1000, actual=500)
    with pytest.raises(DownloadError) as exc_info:
        adapter.download("BVtrunc", "https://www.bilibili.com/video/BVtrunc")
    assert "content-length" in str(exc_info.value).lower() or "1000" in str(exc_info.value)


def test_full_response_succeeds(adapter_against_local_server, tmp_path) -> None:
    """When server declares and returns matching sizes, download() returns the path."""
    adapter, _ = adapter_against_local_server("BVfull", declared=500, actual=500)
    result = adapter.download("BVfull", "https://www.bilibili.com/video/BVfull")
    assert result.exists()
    assert result.stat().st_size == 500