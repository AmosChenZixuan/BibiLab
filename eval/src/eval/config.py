from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from bibilab.config import AIConfig, bibilab_home

from eval.models import ProfileSnapshot

_EVAL_CONFIG_NAME = "eval_config.json"

ProfileName = Literal["generate", "test", "grade"]
PROFILE_NAMES: tuple[ProfileName, ...] = ("generate", "test", "grade")


class Language(str, Enum):
    ZH = "zh"
    EN = "en"

    @property
    def display_name(self) -> str:
        return {"zh": "Chinese", "en": "English"}[self.value]


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

    @field_validator("profiles", mode="before")
    @classmethod
    def _filter_known_profiles(cls, v):
        # Drop unknown profile names so a stale on-disk config never leaks
        # entries that resolve_profile cannot use.
        if not isinstance(v, dict):
            return v
        return {k: val for k, val in v.items() if k in PROFILE_NAMES}

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


def resolve_profile(profile: str) -> AIConfig:
    if profile not in PROFILE_NAMES:
        raise KeyError(f"Unknown eval profile '{profile}'. Valid: {list(PROFILE_NAMES)}")

    entry = load_eval_config().get_profile(profile)

    if entry is None:
        from bibilab.config import load_config
        return load_config().ai

    return AIConfig(
        protocol=entry.protocol,
        model=entry.model,
        api_key=entry.api_key or "ollama",
        base_url=entry.base_url,
    )


def get_language() -> Language:
    return load_eval_config().language


def get_response_language() -> str:
    """Display name used in LLM prompts (e.g. 'Chinese', 'English')."""
    return get_language().display_name
