import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response

from bibilab.config import BibilabConfig, get_config, save_config

log = logging.getLogger(__name__)

router = APIRouter()

BILIBILI_GENERATE_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
BILIBILI_POLL_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
BILIBILI_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"

BILIBILI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.bilibili.com",
}
_BILIBILI_COOKIE_KEYS = {"SESSDATA", "bili_jct", "DedeUserID", "DedeUserID__ckMd5", "sid"}

_QR_WAITING = 86101
_QR_SCANNED = 86090
_QR_EXPIRED = 86038
_QR_SUCCESS = 0


@router.post("/auth/bilibili/qr")
async def generate_qr() -> dict:
    async with httpx.AsyncClient(timeout=10, headers=BILIBILI_HEADERS) as client:
        resp = await client.get(BILIBILI_GENERATE_URL)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to contact Bilibili")
        data = resp.json()
        return {"url": data["data"]["url"], "key": data["data"]["qrcode_key"]}


@router.get("/auth/bilibili/qr/status")
async def qr_status(key: str, cfg: BibilabConfig = Depends(get_config)) -> dict:
    async with httpx.AsyncClient(timeout=10, headers=BILIBILI_HEADERS) as client:
        resp = await client.get(BILIBILI_POLL_URL, params={"qrcode_key": key})
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to contact Bilibili")
    data = resp.json()
    code = data["data"]["code"]

    if code == _QR_WAITING:
        return {"status": "waiting"}
    if code == _QR_SCANNED:
        return {"status": "scanned"}
    if code == _QR_EXPIRED:
        return {"status": "expired"}
    if code == _QR_SUCCESS:
        pairs = urlparse(data["data"]["url"]).query.split("&")
        cookie_str = "; ".join(p for p in pairs if p.split("=", 1)[0] in _BILIBILI_COOKIE_KEYS)
        cfg.accounts.bilibili.cookie = cookie_str
        cfg.accounts.bilibili.last_verified = _iso_now()
        try:
            async with httpx.AsyncClient(timeout=10, headers=BILIBILI_HEADERS) as nav_client:
                nav_resp = await nav_client.get(
                    BILIBILI_NAV_URL,
                    headers={"Cookie": cookie_str},
                )
                if nav_resp.status_code == 200:
                    nav_data = nav_resp.json()
                    cfg.accounts.bilibili.username = nav_data.get("data", {}).get("uname", "")
                    cfg.accounts.bilibili.avatar_url = nav_data.get("data", {}).get("face", "")
        except Exception:
            log.exception("failed to fetch bilibili user info")
        save_config(cfg)
        return {"status": "success"}

    raise HTTPException(status_code=502, detail=f"Unexpected Bilibili code: {code}")


@router.delete("/auth/bilibili")
async def delete_bilibili_auth(cfg: BibilabConfig = Depends(get_config)) -> Response:
    cfg.accounts.bilibili.cookie = ""
    cfg.accounts.bilibili.last_verified = ""
    cfg.accounts.bilibili.username = ""
    cfg.accounts.bilibili.avatar_url = ""
    save_config(cfg)
    return Response(status_code=204)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
