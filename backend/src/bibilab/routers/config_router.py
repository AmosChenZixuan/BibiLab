from typing import Any

from fastapi import APIRouter

from bibilab.config import BibilabConfig, deep_merge, load_config, save_config

router = APIRouter()

_MASKED = "***"


def _mask(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return config dict with sensitive fields masked."""
    masked = dict(cfg)
    accounts = dict(masked.get("accounts", {}))
    bilibili = dict(accounts.get("bilibili", {}))
    if bilibili.get("cookie"):
        bilibili["cookie"] = _MASKED
    accounts["bilibili"] = bilibili
    masked["accounts"] = accounts

    ai = dict(masked.get("ai", {}))
    if ai.get("api_key"):
        ai["api_key"] = _MASKED
    masked["ai"] = ai

    return masked


@router.get("/config")
async def get_config() -> dict:
    cfg = load_config()
    return _mask(cfg.model_dump())


@router.put("/config")
async def put_config(patch: dict[str, Any]) -> dict:
    cfg = load_config()
    merged = deep_merge(cfg.model_dump(), patch)
    new_cfg = BibilabConfig.model_validate(merged)
    save_config(new_cfg)
    return _mask(new_cfg.model_dump())
