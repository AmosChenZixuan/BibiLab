import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def bibilab_home() -> Path:
    return Path.home() / ".bibilab"


def cover_path(source_id: str) -> Path:
    return bibilab_home() / "covers" / f"{source_id}.jpg"


def transcript_path(video_id: str) -> Path:
    return bibilab_home() / "transcripts" / f"{video_id}.txt"


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
    output_language: str = "ui"  # ui | zh | en | ...; "ui" means follow UI language
    # Maximum transcript character limit before truncation (~100K tokens at ~4 chars/token)
    transcript_char_limit: int = 400_000


class TranscriptionConfig(BaseModel):
    engine: str = "faster-whisper"
    model_size: str = "large-v3"
    device: str = "cuda"  # cuda | cpu
    language: str = "auto"  # auto | zh | en
    # LLM call settings (used during summarization/synthesis steps)
    llm_timeout: int = 120
    llm_max_tokens: int = 2048
    # Chunking
    target_tokens: int = 300
    chunk_max_tokens: int = 400
    # Transcription
    beam_size: int = 5


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
    # ChromaDB
    transcript_collection_name: str = "bibilab_transcripts"


_config_cache: BibilabConfig | None = None


def load_config() -> BibilabConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    home = bibilab_home()
    home.mkdir(parents=True, exist_ok=True)
    path = _config_path()
    if not path.exists():
        _config_cache = BibilabConfig()
        return _config_cache
    with path.open() as f:
        data = json.load(f)
    _config_cache = BibilabConfig.model_validate(data)
    return _config_cache


def get_config() -> BibilabConfig:
    """FastAPI dependency that returns the cached config."""
    return load_config()


def save_config(cfg: BibilabConfig) -> None:
    global _config_cache
    path = _config_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(cfg.model_dump_json(indent=2))
    tmp.chmod(0o600)
    os.replace(tmp, path)
    _config_cache = None


_MISSING = object()


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key, _MISSING), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
