import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def bibilab_home() -> Path:
    return Path.home() / ".bibilab"


def resolve_storage_path(stored: str) -> Path:
    return bibilab_home() / stored


def _config_path() -> Path:
    return bibilab_home() / "config.json"


class BilibiliAccountConfig(BaseModel):
    cookie: str = ""
    last_verified: str = ""


class AccountsConfig(BaseModel):
    bilibili: BilibiliAccountConfig = BilibiliAccountConfig()


class AIConfig(BaseModel):
    provider: str = "openai"  # openai | anthropic | ollama | custom
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"


class TranscriptionConfig(BaseModel):
    engine: str = "faster-whisper"
    model_size: str = "large-v3"
    device: str = "cuda"  # cuda | cpu
    language: str = "auto"  # auto | zh | en


class VisionConfig(BaseModel):
    enabled: bool = False
    frame_sample_rate: int = 30
    model: str | None = None


class BackendConfig(BaseModel):
    port: int = 8765
    worker_concurrency: int = 1


class BibilabConfig(BaseModel):
    accounts: AccountsConfig = AccountsConfig()
    ai: AIConfig = AIConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    vision: VisionConfig = VisionConfig()
    backend: BackendConfig = BackendConfig()


def load_config() -> BibilabConfig:
    home = bibilab_home()
    home.mkdir(parents=True, exist_ok=True)
    path = _config_path()
    if not path.exists():
        return BibilabConfig()
    with path.open() as f:
        data = json.load(f)
    return BibilabConfig.model_validate(data)


def save_config(cfg: BibilabConfig) -> None:
    path = _config_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(cfg.model_dump_json(indent=2))
    tmp.chmod(0o600)
    os.replace(tmp, path)


_MISSING = object()


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key, _MISSING), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
