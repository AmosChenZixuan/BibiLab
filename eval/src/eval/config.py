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

VALID_LANGUAGES = ("zh", "en")
DEFAULT_LANGUAGE = "zh"
LANGUAGE_DISPLAY_NAMES = {"zh": "Chinese", "en": "English"}

PROFILE_NAMES = ("generate", "test", "grade")

DEFAULT_EVAL_CONFIG: dict[str, Any] = {
    "profiles": {
        "generate": None,
        "test": {
            "protocol": "openai",
            "model": "glm-4.7-flash",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        },
        "grade": None,
    },
    "language": DEFAULT_LANGUAGE,
}


def _eval_config_path() -> Path:
    return bibilab_home() / _EVAL_CONFIG_NAME


def load_eval_config() -> dict[str, Any]:
    path = _eval_config_path()
    cfg = deepcopy(DEFAULT_EVAL_CONFIG)
    if not path.exists():
        return cfg
    with path.open() as f:
        data = json.load(f)
    if isinstance(data.get("profiles"), dict):
        for name in PROFILE_NAMES:
            if name in data["profiles"]:
                cfg["profiles"][name] = data["profiles"][name]
    if data.get("language") in VALID_LANGUAGES:
        cfg["language"] = data["language"]
    return cfg


def save_eval_config(cfg: dict[str, Any]) -> None:
    path = _eval_config_path()
    with _config_lock:
        tmp = path.with_suffix(f".tmp.{threading.current_thread().name}")
        tmp.write_text(json.dumps(cfg, indent=2))
        tmp.chmod(0o600)
        os.replace(tmp, path)


def resolve_profile(profile: str) -> AIConfig:
    if profile not in PROFILE_NAMES:
        raise KeyError(f"Unknown eval profile '{profile}'. Valid: {list(PROFILE_NAMES)}")

    eval_cfg = load_eval_config()
    entry = eval_cfg["profiles"].get(profile)

    if entry is None:
        from bibilab.config import load_config
        return load_config().ai

    return AIConfig(
        protocol=entry.get("protocol", "openai"),
        model=entry.get("model", ""),
        api_key=entry.get("api_key") or "ollama",
        base_url=entry.get("base_url") or None,
    )


def get_language() -> str:
    return load_eval_config().get("language", DEFAULT_LANGUAGE)


def get_response_language() -> str:
    """Display name used in LLM prompts (e.g. 'Chinese', 'English')."""
    return LANGUAGE_DISPLAY_NAMES[get_language()]
