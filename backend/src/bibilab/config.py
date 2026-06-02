import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


def bibilab_home() -> Path:
    return Path.home() / ".bibilab"


def models_dir(*parts: str) -> Path:
    """Path under ~/.bibilab/models/. Use for model download directories."""
    return bibilab_home().joinpath("models", *parts)


def cover_path(source_id: str) -> Path:
    return bibilab_home() / "covers" / f"{source_id}.jpg"


def _config_path() -> Path:
    return bibilab_home() / "config.json"


class BilibiliAccountConfig(BaseModel):
    cookie: str = ""
    last_verified: str = ""
    username: str = ""
    avatar_url: str = ""


class AccountsConfig(BaseModel):
    bilibili: BilibiliAccountConfig = BilibiliAccountConfig()


class AIConfig(BaseModel):
    protocol: str = "openai"  # openai | anthropic
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    output_language: str = "ui"  # ui | zh | en | ...; "ui" means follow UI language
    # Maximum transcript character limit before truncation (~100K tokens at ~4 chars/token)
    transcript_char_limit: int = 400_000


class TranscriptionConfig(BaseModel):
    model: str = "sensevoice-small"
    device: str = "cuda"  # cuda | cpu
    language: str = "auto"  # auto | zh | en
    llm_timeout: int = 120

    @field_validator("device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {v!r}")
        return v


class VisionConfig(BaseModel):
    enabled: bool = False
    frame_sample_rate: int = 30
    model: str | None = None


class BackendConfig(BaseModel):
    port: int = 8765
    # Max ingest jobs in flight. Governs IO-stage (download + digest LLM call)
    # parallelism only — transcription is serialized by a lock regardless, since
    # it is GPU-compute/GIL-bound and gains nothing from concurrency.
    max_concurrent_jobs: int = 4
    cors_origins: list[str] = [
        "http://localhost",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:5173",
    ]


class RagConfig(BaseModel):
    max_distance: float = 0.8
    reranking_enabled: bool = True
    hybrid_enabled: bool = True
    # Minimum gap (seconds) between Whisper segments to trigger a chunk flush.
    # Pauses longer than this are treated as topic boundaries. Default 1.5s.
    # Keep in sync with pause_threshold_seconds default in pipeline/chunk.py.
    chunk_pause_threshold: float = Field(default=1.5, gt=0)
    # Opt-in: dump per-turn LLM input to ~/.bibilab/debug/{message_id}.json (#393).
    debug_prompts: bool = False


class BibilabConfig(BaseModel):
    accounts: AccountsConfig = AccountsConfig()
    ai: AIConfig = AIConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    vision: VisionConfig = VisionConfig()
    backend: BackendConfig = BackendConfig()
    rag: RagConfig = RagConfig()
    # ChromaDB
    transcript_collection_name: str = "bibilab_transcripts"


_config_cache: BibilabConfig | None = None
_config_lock = threading.Lock()


def _reset_cache() -> None:
    """Reset the config cache. For testing only."""
    global _config_cache
    with _config_lock:
        _config_cache = None


def load_config() -> BibilabConfig:
    global _config_cache
    with _config_lock:
        if _config_cache is not None:
            return _config_cache.model_copy(deep=True)
        home = bibilab_home()
        home.mkdir(parents=True, exist_ok=True)
        path = _config_path()
        if not path.exists():
            _config_cache = BibilabConfig()
            return _config_cache.model_copy(deep=True)
        with path.open() as f:
            data = json.load(f)
        _config_cache = BibilabConfig.model_validate(data)
        return _config_cache.model_copy(deep=True)


def get_config() -> BibilabConfig:
    """FastAPI dependency that returns the cached config."""
    return load_config()


def save_config(cfg: BibilabConfig) -> None:
    global _config_cache
    path = _config_path()
    # Snapshot cfg inside lock to avoid storing caller's mutable ref
    with _config_lock:
        cfg_snapshot = cfg.model_copy(deep=True)
        # Use unique temp file per thread to avoid concurrent write collisions
        tmp = path.with_suffix(f".{threading.current_thread().name}.tmp")
        tmp.write_text(cfg_snapshot.model_dump_json(indent=2))
        tmp.chmod(0o600)
        os.replace(tmp, path)
        _config_cache = cfg_snapshot


_MISSING = object()


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key, _MISSING), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
