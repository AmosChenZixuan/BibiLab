import httpx
from fastapi import APIRouter, Depends, HTTPException, Response

from bibilab.config import BibilabConfig, get_config, save_config

router = APIRouter()

BILIBILI_GENERATE_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
BILIBILI_POLL_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"


@router.post("/auth/bilibili/qr")
async def generate_qr(cfg: BibilabConfig = Depends(get_config)) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(BILIBILI_GENERATE_URL)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to contact Bilibili")
    data = resp.json()
    return {"url": data["data"]["url"], "key": data["data"]["qrcode_key"]}


@router.get("/auth/bilibili/qr/{key}/status")
async def qr_status(key: str, cfg: BibilabConfig = Depends(get_config)) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(BILIBILI_POLL_URL, params={"qrcode_key": key})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to contact Bilibili")
    data = resp.json()
    code = data["data"]["code"]

    if code == 86101:
        return {"status": "waiting"}
    if code == 86090:
        return {"status": "scanned"}
    if code == 86038:
        return {"status": "expired"}
    if code == 0:
        cookies = resp.headers.get_list("set-cookie")
        cookie_str = "; ".join(c.split(";", 1)[0] for c in cookies if c)
        cfg.accounts.bilibili.cookie = cookie_str
        cfg.accounts.bilibili.last_verified = _iso_now()
        save_config(cfg)
        return {"status": "success"}

    raise HTTPException(status_code=502, detail=f"Unexpected Bilibili code: {code}")


@router.delete("/auth/bilibili")
async def delete_bilibili_auth(cfg: BibilabConfig = Depends(get_config)) -> Response:
    cfg.accounts.bilibili.cookie = ""
    cfg.accounts.bilibili.last_verified = ""
    save_config(cfg)
    return Response(status_code=204)


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
