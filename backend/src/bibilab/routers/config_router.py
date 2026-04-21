from typing import Any

from fastapi import APIRouter, Depends

from bibilab.config import BibilabConfig, deep_merge, get_config, save_config

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
async def get_config_handler(cfg: BibilabConfig = Depends(get_config)) -> dict:
    return _mask(cfg.model_dump())


@router.put("/config")
async def put_config(patch: dict[str, Any], cfg: BibilabConfig = Depends(get_config)) -> dict:
    _unmask_patch(patch)
    merged = deep_merge(cfg.model_dump(), patch)
    new_cfg = BibilabConfig.model_validate(merged)
    save_config(new_cfg)
    return _mask(new_cfg.model_dump())


def _unmask_patch(patch: dict[str, Any]) -> None:
    """Remove masked sentinel values from patch so they don't overwrite real values."""
    ai = patch.get("ai")
    if ai and ai.get("api_key") == _MASKED:
        ai.pop("api_key", None)
    accounts = patch.get("accounts")
    if accounts:
        bilibili = accounts.get("bilibili")
        if bilibili and bilibili.get("cookie") == _MASKED:
            bilibili.pop("cookie", None)
