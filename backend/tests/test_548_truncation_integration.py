"""#548 AC3 (integration): a server that returns fewer bytes than Content-Length
must surface as DownloadError — no silent short file.

Runs a tiny aiohttp server in-process that serves a body shorter than the
declared Content-Length. Skipped from the unit lane via the integration marker.
"""

import asyncio
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class _TruncatingHandler:
    """aiohttp handler: declares Content-Length=N, writes only M < N bytes."""

    def __init__(self, declared_length: int, actual_length: int) -> None:
        self.declared_length = declared_length
        self.actual_length = actual_length
        self.received_requests = 0

    async def handle(self, request):
        from aiohttp import web

        self.received_requests += 1
        body = b"x" * self.actual_length
        return web.Response(
            body=body,
            headers={"Content-Length": str(self.declared_length)},
        )


def _start_server(handler: _TruncatingHandler) -> tuple[str, callable]:
    """Spin up an aiohttp app on a free port; return (base_url, stop_fn)."""
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/file.bin", handler.handle)
    runner = web.AppRunner(app)

    async def _run():
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        # Discover the actual port chosen.
        return runner.addresses[0][1]

    loop = asyncio.new_event_loop()
    port_holder: list[int] = []

    def _runner():
        asyncio.set_event_loop(loop)
        port_holder.append(loop.run_until_complete(_run()))
        loop.run_forever()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    # Wait for port discovery.
    for _ in range(50):
        if port_holder:
            break
        threading.Event().wait(0.05)
    port = port_holder[0]

    def stop():
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)

    return f"http://127.0.0.1:{port}/file.bin", stop


def test_truncated_response_raises_download_error(tmp_path: Path) -> None:
    """AC3: declared Content-Length=1000, server returns 500 → DownloadError."""
    from bibilab.adapters.base import DownloadError
    from bibilab.adapters.bilibili import BilibiliAdapter

    # Mock yt-dlp resolve to point at our truncating server.
    handler = _TruncatingHandler(declared_length=1000, actual_length=500)
    url, stop = _start_server(handler)
    try:
        info = {
            "url": url,
            "ext": "m4a",
            "protocol": "https",
            "fragments": None,
            "http_headers": {},
            "filesize": 1000,
        }
        adapter = BilibiliAdapter(cookie="")

        from bibilab.config import _reset_cache, get_config  # noqa: F401

        with pytest.MonkeyPatch.context() as mp:
            # Pin a clean config cache.
            from bibilab.config import load_config as _lc
            _lc()
            mp.setattr("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", _FakeYDL(info))
            mp.setattr("bibilab.adapters.bilibili.bibilab_home", lambda: tmp_path)
            with pytest.raises(DownloadError) as exc_info:
                adapter.download("BVtrunc", "https://www.bilibili.com/video/BVtrunc")

        assert "content-length" in str(exc_info.value).lower() or "1000" in str(exc_info.value)
        # No short file should remain at downloads/BVtrunc.*.
        leftover = list((tmp_path / "downloads").glob("BVtrunc.*")) if (tmp_path / "downloads").exists() else []
        # Either no leftover, or any leftover is the full 1000 bytes (not 500).
        for f in leftover:
            assert f.stat().st_size == 1000, f"short file remains: {f}"
    finally:
        stop()


def test_full_response_succeeds(tmp_path: Path) -> None:
    """AC1 (integration sanity): when server declares and returns matching
    sizes, download() returns the path with the full size on disk."""
    from bibilab.adapters.bilibili import BilibiliAdapter

    handler = _TruncatingHandler(declared_length=500, actual_length=500)
    url, stop = _start_server(handler)
    try:
        info = {
            "url": url,
            "ext": "m4a",
            "protocol": "https",
            "fragments": None,
            "http_headers": {},
            "filesize": 500,
        }
        adapter = BilibiliAdapter(cookie="")
        with pytest.MonkeyPatch.context() as mp:
            from bibilab.config import load_config as _lc
            _lc()
            mp.setattr("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", _FakeYDL(info))
            mp.setattr("bibilab.adapters.bilibili.bibilab_home", lambda: tmp_path)
            result = adapter.download("BVfull", "https://www.bilibili.com/video/BVfull")
        assert result.exists()
        assert result.stat().st_size == 500
    finally:
        stop()


class _FakeYDL:
    """Mock yt_dlp.YoutubeDL returning the fixed info dict for resolve()."""

    def __init__(self, info: dict) -> None:
        self.info = info

    def __call__(self, opts):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=False):
        return self.info