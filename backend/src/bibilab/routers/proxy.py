from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter()

ALLOWED_DOMAINS = {"hdslb.com"}
MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5MB


@router.get("/proxy/cover")
async def proxy_cover(url: str = Query(..., description="URL to proxy")):
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    if host not in ALLOWED_DOMAINS and not any(host.endswith("." + d) for d in ALLOWED_DOMAINS):
        raise HTTPException(status_code=400, detail="URL domain not allowed")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.bilibili.com/",
    }

    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        try:
            resp = await client.get(url, headers=headers)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch: {exc}")

    if resp.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"Upstream returned {resp.status_code}")

    if len(resp.content) > MAX_RESPONSE_SIZE:
        raise HTTPException(status_code=502, detail="Response too large (max 5MB)")

    content_type = resp.headers.get("content-type", "image/jpeg")
    return Response(content=resp.content, media_type=content_type)
