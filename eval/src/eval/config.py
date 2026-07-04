from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from eval.models import ProfileSnapshot

_EVAL_CONFIG_NAME = "eval_config.json"

DEFAULT_BACKEND_URL = "http://127.0.0.1:8765"


def bibilab_home() -> Path:
    """Same resolution as the backend's (env override, else ~/.bibilab) —
    only for eval's own files (eval_config.json, evals/ storage); all backend
    data is reached over HTTP."""
    return Path(os.environ.get("BIBILAB_HOME", "~/.bibilab")).expanduser()

ProfileName = Literal["generate", "test", "grade"]
PROFILE_NAMES: tuple[ProfileName, ...] = ("generate", "test", "grade")


class Language(str, Enum):
    ZH = "zh"
    EN = "en"


DEFAULT_LANGUAGE = Language.ZH


class EvalConfig(BaseModel):
    profiles: dict[str, ProfileSnapshot | None] = Field(
        default_factory=lambda: {
            "generate": None,
            "test": ProfileSnapshot(
                protocol="openai",
                model="glm-4.7-flash",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            "grade": None,
        }
    )
    language: Language = DEFAULT_LANGUAGE
    backend_url: str = DEFAULT_BACKEND_URL

    @field_validator("profiles", mode="before")
    @classmethod
    def _filter_known_profiles(cls, v):
        # Drop unknown profile names so a stale on-disk config never leaks
        # entries that resolve_profile cannot use.
        if not isinstance(v, dict):
            return v
        return {k: val for k, val in v.items() if k in PROFILE_NAMES}

    @field_validator("backend_url", mode="before")
    @classmethod
    def _fallback_backend_url(cls, v):
        # Blank falls back to default rather than persisting: an empty base_url
        # makes httpx raise on every call, and the TUI can't render past it to
        # let the user fix it.
        if isinstance(v, str):
            v = v.strip()
        return v or DEFAULT_BACKEND_URL

    @field_validator("language", mode="before")
    @classmethod
    def _fallback_language(cls, v):
        # Unknown language values fall back to default rather than raising — the
        # config UI may have written a typo, and a stale value should never break
        # the eval CLI.
        try:
            return Language(v)
        except (ValueError, KeyError):
            return DEFAULT_LANGUAGE

    def get_profile(self, name: str) -> ProfileSnapshot | None:
        return self.profiles.get(name)


def _eval_config_path() -> Path:
    return bibilab_home() / _EVAL_CONFIG_NAME


def load_eval_config() -> EvalConfig:
    path = _eval_config_path()
    if not path.exists():
        return EvalConfig()
    with path.open() as f:
        data = json.load(f)
    return EvalConfig.model_validate(data)


def save_eval_config(cfg: EvalConfig) -> None:
    path = _eval_config_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(cfg.model_dump_json(indent=2))
    tmp.chmod(0o600)
    os.replace(tmp, path)


def resolve_profile(profile: str) -> ProfileSnapshot | None:
    """None = no override: requests omit the `llm` field and the backend
    serves the call with its own configured LLM."""
    if profile not in PROFILE_NAMES:
        raise KeyError(f"Unknown eval profile '{profile}'. Valid: {list(PROFILE_NAMES)}")

    return load_eval_config().get_profile(profile)


def get_backend_url() -> str:
    return load_eval_config().backend_url


def get_language() -> Language:
    return load_eval_config().language


def get_response_language() -> str:
    """Language code for the configured response language (e.g. 'zh', 'en').

    Returns the language code, not the display name. bibilab's
    build_grounding_prompt looks this up in _LANG_NATIVE_NAME (which is keyed
    on codes); a display-name string would silently fall through to the
    English default and the chat would answer in the wrong language.
    """
    return get_language().value
