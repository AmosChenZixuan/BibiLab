from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


def _make_mock_response(json_data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json = MagicMock(return_value=json_data)
    return mock_resp


@contextmanager
def _patch_bilibili_httpx(mock_resp):
    mock_http_client = MagicMock()
    mock_http_client.get = AsyncMock(return_value=mock_resp)
    mock_http_client.aclose = AsyncMock()

    with patch("bibilab.routers.auth.httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=mock_http_client)
        instance.__aexit__ = AsyncMock()
        mock_client_cls.return_value = instance
        yield


@pytest.mark.asyncio
async def test_generate_qr_returns_url_and_key(client: httpx.AsyncClient):
    mock_resp = _make_mock_response(
        {
            "data": {
                "url": "https://bilibili.com/qr/abc123",
                "qrcode_key": "key-abc123",
            }
        }
    )

    with _patch_bilibili_httpx(mock_resp):
        resp = await client.post("/auth/bilibili/qr")

    assert resp.status_code == 200
    data = resp.json()
    assert data["url"] == "https://bilibili.com/qr/abc123"
    assert data["key"] == "key-abc123"


@pytest.mark.asyncio
async def test_qr_status_waiting(client: httpx.AsyncClient):
    mock_resp = _make_mock_response({"data": {"code": 86101}})

    with _patch_bilibili_httpx(mock_resp):
        resp = await client.get("/auth/bilibili/qr/status?key=some-key")

    assert resp.status_code == 200
    assert resp.json()["status"] == "waiting"


@pytest.mark.asyncio
async def test_qr_status_scanned(client: httpx.AsyncClient):
    mock_resp = _make_mock_response({"data": {"code": 86090}})

    with _patch_bilibili_httpx(mock_resp):
        resp = await client.get("/auth/bilibili/qr/status?key=some-key")

    assert resp.status_code == 200
    assert resp.json()["status"] == "scanned"


@pytest.mark.asyncio
async def test_qr_status_expired(client: httpx.AsyncClient):
    mock_resp = _make_mock_response({"data": {"code": 86038}})

    with _patch_bilibili_httpx(mock_resp):
        resp = await client.get("/auth/bilibili/qr/status?key=some-key")

    assert resp.status_code == 200
    assert resp.json()["status"] == "expired"


@pytest.mark.asyncio
async def test_qr_status_success_saves_cookie(client: httpx.AsyncClient, tmp_bibilab_home):
    mock_resp = _make_mock_response(
        {
            "data": {
                "code": 0,
                "url": "https://passport.bilibili.com/crossDomain?SESSDATA=abc%2C1234567890%2Cxyz&bili_jct=def456&DedeUserID=999&gourl=https%3A%2F%2Fwww.bilibili.com",
            }
        }
    )

    with _patch_bilibili_httpx(mock_resp):
        resp = await client.get("/auth/bilibili/qr/status?key=some-key")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    from bibilab.config import load_config

    cfg = load_config()
    assert "SESSDATA=abc%2C1234567890%2Cxyz" in cfg.accounts.bilibili.cookie
    assert "bili_jct=def456" in cfg.accounts.bilibili.cookie
    assert "gourl" not in cfg.accounts.bilibili.cookie
    assert cfg.accounts.bilibili.last_verified != ""


@contextmanager
def _patch_bilibili_httpx_named(side_effect_fn):
    """Patch httpx.AsyncClient to return different responses per URL."""
    mock_http_client = MagicMock()

    async def mock_get(url, **kwargs):
        resp = side_effect_fn(str(url), **kwargs)
        return resp

    mock_http_client.get = mock_get
    mock_http_client.aclose = AsyncMock()

    with patch("bibilab.routers.auth.httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=mock_http_client)
        instance.__aexit__ = AsyncMock()
        mock_client_cls.return_value = instance
        yield


@pytest.mark.asyncio
async def test_qr_status_success_saves_username_and_avatar(client: httpx.AsyncClient, tmp_bibilab_home):
    poll_mock_resp = _make_mock_response(
        {
            "data": {
                "code": 0,
                "url": "https://passport.bilibili.com/crossDomain?SESSDATA=abc%2C123",
            }
        }
    )
    nav_mock_resp = _make_mock_response(
        {
            "data": {
                "uname": "test_user",
                "face": "https://i0.hdslb.com/bfs/face/abc.jpg",
            }
        }
    )

    def side_effect(url, **kwargs):
        if "/qrcode/poll" in url:
            return poll_mock_resp
        if "/x/web-interface/nav" in url:
            return nav_mock_resp
        return _make_mock_response({})

    with _patch_bilibili_httpx_named(side_effect):
        resp = await client.get("/auth/bilibili/qr/status?key=some-key")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    from bibilab.config import load_config

    cfg = load_config()
    assert cfg.accounts.bilibili.username == "test_user"
    assert cfg.accounts.bilibili.avatar_url == "https://i0.hdslb.com/bfs/face/abc.jpg"
    assert "SESSDATA=abc%2C123" in cfg.accounts.bilibili.cookie


@pytest.mark.asyncio
async def test_qr_status_success_nav_failure_does_not_clear_cookie(client: httpx.AsyncClient, tmp_bibilab_home):
    poll_mock_resp = _make_mock_response(
        {
            "data": {
                "code": 0,
                "url": "https://passport.bilibili.com/crossDomain?SESSDATA=abc%2C123",
            }
        }
    )
    nav_mock_resp = _make_mock_response({}, status_code=500)

    def side_effect(url, **kwargs):
        if "/qrcode/poll" in url:
            return poll_mock_resp
        if "/x/web-interface/nav" in url:
            return nav_mock_resp
        return _make_mock_response({})

    with _patch_bilibili_httpx_named(side_effect):
        resp = await client.get("/auth/bilibili/qr/status?key=some-key")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    from bibilab.config import load_config

    cfg = load_config()
    assert "SESSDATA=abc%2C123" in cfg.accounts.bilibili.cookie
    assert cfg.accounts.bilibili.username == ""
    assert cfg.accounts.bilibili.avatar_url == ""


@pytest.mark.asyncio
async def test_delete_bilibili_auth_clears_all_fields(client: httpx.AsyncClient, tmp_bibilab_home):
    from bibilab.config import load_config, save_config

    cfg = load_config()
    cfg.accounts.bilibili.cookie = "SESSDATA=old"
    cfg.accounts.bilibili.last_verified = "2025-01-01T00:00:00Z"
    cfg.accounts.bilibili.username = "old_user"
    cfg.accounts.bilibili.avatar_url = "https://i0.hdslb.com/bfs/face/old.jpg"
    save_config(cfg)

    resp = await client.delete("/auth/bilibili")
    assert resp.status_code == 204

    cfg = load_config()
    assert cfg.accounts.bilibili.cookie == ""
    assert cfg.accounts.bilibili.last_verified == ""
    assert cfg.accounts.bilibili.username == ""
    assert cfg.accounts.bilibili.avatar_url == ""
