from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_proxy_cover_rejects_invalid_domain(client: httpx.AsyncClient):
    response = await client.get("/proxy/cover?url=https://evil.com/image.jpg")
    assert response.status_code == 400
    assert "domain not allowed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_proxy_cover_rejects_www_non_allowed_domain(client: httpx.AsyncClient):
    response = await client.get("/proxy/cover?url=https://www.evil.com/img.jpg")
    assert response.status_code == 400
    assert "domain not allowed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_proxy_cover_allows_hdslb_domain(client: httpx.AsyncClient):
    with patch("bibilab.routers.proxy.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"fake-image-data"

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        response = await client.get("/proxy/cover?url=https://i0.hdslb.com/bfs/archive/img.jpg")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_proxy_cover_rejects_domain_bypass(client: httpx.AsyncClient):
    response = await client.get("/proxy/cover?url=https://evilbilibili.com/img.jpg")
    assert response.status_code == 400
    assert "domain not allowed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_proxy_cover_allows_subdomain(client: httpx.AsyncClient):
    with patch("bibilab.routers.proxy.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"fake-image-data"

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        response = await client.get("/proxy/cover?url=https://i1.hdslb.com/bfs/archive/img.jpg")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_proxy_cover_allows_ytimg_without_referer(client: httpx.AsyncClient):
    with patch("bibilab.routers.proxy.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"fake-image-data"

        mock_get = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.get = mock_get

        response = await client.get("/proxy/cover?url=https://i.ytimg.com/vi/x/hqdefault.jpg")
        assert response.status_code == 200
        sent_headers = mock_get.call_args.kwargs["headers"]
        assert "Referer" not in sent_headers


@pytest.mark.asyncio
async def test_proxy_cover_hdslb_sends_bilibili_referer(client: httpx.AsyncClient):
    with patch("bibilab.routers.proxy.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"fake-image-data"

        mock_get = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.get = mock_get

        response = await client.get("/proxy/cover?url=https://i0.hdslb.com/bfs/archive/img.jpg")
        assert response.status_code == 200
        sent_headers = mock_get.call_args.kwargs["headers"]
        assert sent_headers["Referer"] == "https://www.bilibili.com/"


@pytest.mark.asyncio
async def test_proxy_cover_rejects_ytimg_suffix_spoof(client: httpx.AsyncClient):
    response = await client.get("/proxy/cover?url=https://evilytimg.com/img.jpg")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_proxy_cover_rejects_oversized_response(client: httpx.AsyncClient):
    large_content = b"x" * (6 * 1024 * 1024)  # 6MB
    with patch("bibilab.routers.proxy.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = large_content

        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        response = await client.get("/proxy/cover?url=https://i0.hdslb.com/bfs/archive/img.jpg")
        assert response.status_code == 502
        assert "too large" in response.json()["detail"]


@pytest.mark.asyncio
async def test_proxy_cover_does_not_follow_redirect_to_malicious_host(client: httpx.AsyncClient):
    """A 302 redirect to a malicious host (169.254.169.254) should be rejected, not followed."""
    with patch("bibilab.routers.proxy.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 302
        mock_response.headers = {"location": "http://169.254.169.254/latest/meta-data/"}

        mock_get = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.get = mock_get

        response = await client.get("/proxy/cover?url=https://i0.hdslb.com/bfs/archive/img.jpg")
        assert response.status_code == 502
        assert "Upstream returned 302" in response.json()["detail"]
