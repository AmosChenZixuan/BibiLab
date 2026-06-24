import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


def bibilab_home() -> Path:
    override = os.environ.get("BIBILAB_HOME")
    if override:
        return Path(override)
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
    # Total token window of the configured model (input + output). Used by
    # resolve_max_tokens as the input-side fail-loud threshold. Default 128K
    # covers most modern cloud and local models; lower it for small local
    # models to keep input + reserved output inside their real window.
    context_window: int = 128000
    # Per-call output budget. User-chosen from the LLM tab's 4-position
    # slot slider (16K / 32K / 64K / 100K). Default 16K matches the
    # observed upper bound on thinking + answer for all real tasks (digest,
    # chat, eval generate). See resolve_max_tokens for the overflow check.
    max_output_tokens: int = 16384

    @model_validator(mode="after")
    def _output_fits_in_window(self) -> "AIConfig":
        if self.max_output_tokens >= self.context_window:
            raise ValueError(
                f"max_output_tokens ({self.max_output_tokens}) must be less than "
                f"context_window ({self.context_window})."
            )
        return self


class TranscriptionConfig(BaseModel):
    model: str = "sensevoice-small"
    device: str = "cuda"  # cuda | cpu
    language: str = "auto"  # auto | zh | en

    @field_validator("device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {v!r}")
        return v


class BackendConfig(BaseModel):
    port: int = 8765
    # Max ingest jobs in flight. Governs IO-stage (download + digest LLM call)
    # parallelism only — transcription is serialized by a lock regardless, since
    # it is GPU-compute/GIL-bound and gains nothing from concurrency.
    max_concurrent_jobs: int = 4
    # Per-file connection count fed to aria2c as `-x{n} -s{n}`. Bounds the
    # per-IP throttle tail on single/low-concurrency ingests where per-file
    # connections are the only parallel dimension; ignored on hosts without
    # aria2c on PATH.
    download_connections: int = 16
    cors_origins: list[str] = [
        "http://localhost",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:5173",
    ]

    @model_validator(mode="before")
    @classmethod
    def _check_download_connections_and_scale(cls, data: Any) -> Any:
        # Two responsibilities, in order: bounds check first (fail loud),
        # then auto-scale so the per-IP budget product (max_concurrent_jobs
        # × download_connections) stays ≤ 64. The bench ceiling is ~20 MB/s
        # aggregate; past ~64 sub-conns the per-connection throttle this
        # whole knob exists to bound comes back via aggregate load.
        # Single-video (jobs=1) keeps the bench-calibrated 16.
        # mode="before" (mutates the input dict) so this fires on both
        # __init__ and JSON-deserialize paths; mode="after" returning
        # model_copy is silently ignored when validating via __init__.
        if isinstance(data, dict):
            conns = data.get("download_connections", 16)
            if not 1 <= conns <= 64:
                raise ValueError(f"download_connections must be in [1, 64], got {conns!r}")
            jobs = data.get("max_concurrent_jobs", 4)
            if jobs * conns > 64:
                data = {**data, "download_connections": max(1, 64 // jobs)}
        return data


class RagConfig(BaseModel):
    max_distance: float = 0.8
    reranking_enabled: bool = True
    hybrid_enabled: bool = True
    # Opt-in: dump one JSON per chat turn to ~/.bibilab/debug/{message_id}.json,
    # capturing the final cumulative LLM state (system, tools, messages, response, model, timestamp).
    debug_prompts: bool = False


class BibilabConfig(BaseModel):
    accounts: AccountsConfig = AccountsConfig()
    ai: AIConfig = AIConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    backend: BackendConfig = BackendConfig()
    rag: RagConfig = RagConfig()


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
