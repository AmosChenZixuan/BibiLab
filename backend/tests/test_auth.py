from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


def _make_mock_response(json_data, status_code=200, cookie_headers=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json = MagicMock(return_value=json_data)
    headers_mock = MagicMock()
    headers_mock.get_list.return_value = cookie_headers or []
    mock_resp.headers = headers_mock
    return mock_resp


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

    mock_http_client = MagicMock()
    mock_http_client.post = AsyncMock(return_value=mock_resp)
    mock_http_client.aclose = AsyncMock()

    with patch("bibilab.routers.auth.httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=mock_http_client)
        instance.__aexit__ = AsyncMock()
        mock_client_cls.return_value = instance

        resp = await client.post("/auth/bilibili/qr")

    assert resp.status_code == 200
    data = resp.json()
    assert data["url"] == "https://bilibili.com/qr/abc123"
    assert data["key"] == "key-abc123"


@pytest.mark.asyncio
async def test_qr_status_waiting(client: httpx.AsyncClient):
    mock_resp = _make_mock_response({"data": {"code": 86101}})

    mock_http_client = MagicMock()
    mock_http_client.get = AsyncMock(return_value=mock_resp)
    mock_http_client.aclose = AsyncMock()

    with patch("bibilab.routers.auth.httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=mock_http_client)
        instance.__aexit__ = AsyncMock()
        mock_client_cls.return_value = instance

        resp = await client.get("/auth/bilibili/qr/some-key/status")

    assert resp.status_code == 200
    assert resp.json()["status"] == "waiting"


@pytest.mark.asyncio
async def test_qr_status_scanned(client: httpx.AsyncClient):
    mock_resp = _make_mock_response({"data": {"code": 86090}})

    mock_http_client = MagicMock()
    mock_http_client.get = AsyncMock(return_value=mock_resp)
    mock_http_client.aclose = AsyncMock()

    with patch("bibilab.routers.auth.httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=mock_http_client)
        instance.__aexit__ = AsyncMock()
        mock_client_cls.return_value = instance

        resp = await client.get("/auth/bilibili/qr/some-key/status")

    assert resp.status_code == 200
    assert resp.json()["status"] == "scanned"


@pytest.mark.asyncio
async def test_qr_status_expired(client: httpx.AsyncClient):
    mock_resp = _make_mock_response({"data": {"code": 86038}})

    mock_http_client = MagicMock()
    mock_http_client.get = AsyncMock(return_value=mock_resp)
    mock_http_client.aclose = AsyncMock()

    with patch("bibilab.routers.auth.httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=mock_http_client)
        instance.__aexit__ = AsyncMock()
        mock_client_cls.return_value = instance

        resp = await client.get("/auth/bilibili/qr/some-key/status")

    assert resp.status_code == 200
    assert resp.json()["status"] == "expired"


@pytest.mark.asyncio
async def test_qr_status_success_saves_cookie(client: httpx.AsyncClient, tmp_bibilab_home):
    mock_resp = _make_mock_response(
        {"data": {"code": 0}}, cookie_headers=["SESSDATA=abc123; Path=/", "BILI_JCT=def456; Path=/"]
    )

    mock_http_client = MagicMock()
    mock_http_client.get = AsyncMock(return_value=mock_resp)
    mock_http_client.aclose = AsyncMock()

    with patch("bibilab.routers.auth.httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=mock_http_client)
        instance.__aexit__ = AsyncMock()
        mock_client_cls.return_value = instance

        resp = await client.get("/auth/bilibili/qr/some-key/status")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    from bibilab.config import load_config

    cfg = load_config()
    assert "SESSDATA=abc123" in cfg.accounts.bilibili.cookie
    assert "BILI_JCT=def456" in cfg.accounts.bilibili.cookie
    assert cfg.accounts.bilibili.last_verified != ""


@pytest.mark.asyncio
async def test_delete_bilibili_auth_clears_cookie(client: httpx.AsyncClient, tmp_bibilab_home):
    from bibilab.config import load_config, save_config

    cfg = load_config()
    cfg.accounts.bilibili.cookie = "SESSDATA=old"
    cfg.accounts.bilibili.last_verified = "2025-01-01T00:00:00Z"
    save_config(cfg)

    resp = await client.delete("/auth/bilibili")
    assert resp.status_code == 204

    cfg = load_config()
    assert cfg.accounts.bilibili.cookie == ""
    assert cfg.accounts.bilibili.last_verified == ""
