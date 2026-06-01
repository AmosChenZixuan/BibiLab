from typing import Any

from fastapi import APIRouter, Depends

from bibilab.config import BibilabConfig, deep_merge, get_config, save_config

router = APIRouter()


def _mask_sensitive(value: str) -> str:
    """Return a partially masked version of a sensitive string for display.
    Shows first 4 and last 4 characters, e.g. 'sk-6e...4bxO'.
    Falls back to full mask for very short strings.
    """
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def _mask(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return config dict with sensitive fields masked."""
    masked = dict(cfg)
    ai = dict(masked.get("ai", {}))
    if ai.get("api_key"):
        ai["api_key"] = _mask_sensitive(ai["api_key"])
    masked["ai"] = ai
    accounts = dict(masked.get("accounts", {}))
    bilibili = dict(accounts.get("bilibili", {}))
    if bilibili.get("cookie"):
        bilibili["cookie"] = _mask_sensitive(bilibili["cookie"])
    accounts["bilibili"] = bilibili
    masked["accounts"] = accounts
    return masked


def _strip_strings(d: dict[str, Any]) -> None:
    """Recursively strip leading/trailing whitespace from all string values."""
    for k, v in d.items():
        if isinstance(v, str):
            d[k] = v.strip()
        elif isinstance(v, dict):
            _strip_strings(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str):
                    v[i] = item.strip()
                elif isinstance(item, dict):
                    _strip_strings(item)


def _is_masked(value: str) -> bool:
    """Check if a value is a masked sentinel (contains '...' as mask indicator).
    Real API keys and cookies never contain '...' naturally.
    """
    return "..." in value


@router.get("/config")
async def get_config_handler(cfg: BibilabConfig = Depends(get_config)) -> dict:
    return _mask(cfg.model_dump())


@router.put("/config")
async def put_config(patch: dict[str, Any], cfg: BibilabConfig = Depends(get_config)) -> dict:
    _strip_strings(patch)
    _unmask_patch(patch)
    merged = deep_merge(cfg.model_dump(), patch)
    new_cfg = BibilabConfig.model_validate(merged)
    save_config(new_cfg)
    return _mask(new_cfg.model_dump())


def _unmask_patch(patch: dict[str, Any]) -> None:
    """Remove masked sentinel values from patch so they don't overwrite real values."""
    ai = patch.get("ai")
    if ai and isinstance(ai.get("api_key"), str) and _is_masked(ai["api_key"]):
        ai.pop("api_key", None)
    accounts = patch.get("accounts")
    if accounts:
        bilibili = accounts.get("bilibili")
        if bilibili and isinstance(bilibili.get("cookie"), str) and _is_masked(bilibili["cookie"]):
            bilibili.pop("cookie", None)
