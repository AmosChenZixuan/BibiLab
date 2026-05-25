from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from bibilab.config import AIConfig, bibilab_home

_EVAL_CONFIG_NAME = "eval_config.json"
_config_lock = threading.Lock()

DEFAULT_EVAL_CONFIG: dict[str, dict[str, Any]] = {
    "generate": {"source": "inherit"},
    "test": {
        "source": "custom",
        "protocol": "openai",
        "model": "glm-4.7-flash",
        "base_url": "http://localhost:11434/v1",
    },
    "grade": {"source": "inherit"},
}


def _eval_config_path() -> Path:
    return bibilab_home() / _EVAL_CONFIG_NAME


def load_eval_config() -> dict[str, dict[str, Any]]:
    path = _eval_config_path()
    if not path.exists():
        return deepcopy(DEFAULT_EVAL_CONFIG)
    with path.open() as f:
        data = json.load(f)
    merged = deepcopy(DEFAULT_EVAL_CONFIG)
    _deep_merge(merged, data)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_eval_config(cfg: dict[str, dict[str, Any]]) -> None:
    path = _eval_config_path()
    with _config_lock:
        tmp = path.with_suffix(f".tmp.{threading.current_thread().name}")
        tmp.write_text(json.dumps(cfg, indent=2))
        tmp.chmod(0o600)
        os.replace(tmp, path)


def resolve_profile(profile: str) -> AIConfig:
    eval_cfg = load_eval_config()
    entry = eval_cfg.get(profile)
    if entry is None:
        valid = list(DEFAULT_EVAL_CONFIG.keys())
        raise KeyError(f"Unknown eval profile '{profile}'. Valid: {valid}")

    source = entry.get("source", "inherit")
    if source == "inherit":
        from bibilab.config import load_config

        backend_cfg = load_config()
        return backend_cfg.ai

    return AIConfig(
        protocol=entry.get("protocol", "openai"),
        model=entry.get("model", ""),
        api_key=entry.get("api_key") or "ollama",
        base_url=entry.get("base_url", ""),
    )
