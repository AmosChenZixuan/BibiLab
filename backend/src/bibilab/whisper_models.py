import shutil
from pathlib import Path

import faster_whisper

from bibilab.config import bibilab_home

SUPPORTED_WHISPER_MODELS: tuple[str, ...] = tuple(faster_whisper.available_models())


def whisper_model_dir() -> Path:
    return bibilab_home() / "models" / "whisper"


def _hf_cache_dir() -> Path:
    return bibilab_home() / ".cache" / "huggingface"


def _download_target_dir(model_size: str) -> Path:
    return whisper_model_dir() / model_size


def _cleanup_download_artifacts(target_dir: Path) -> None:
    shutil.rmtree(target_dir / ".cache", ignore_errors=True)


def _candidate_model_paths(model_size: str) -> list[Path]:
    root = whisper_model_dir()
    legacy_name = root / "whisper" / f"whisper-{model_size}"
    direct_name = root / model_size
    repo_name = root / f"faster-whisper-{model_size}"
    return [legacy_name, direct_name, repo_name]


def resolve_local_model_path(model_size: str) -> Path | None:
    for path in _candidate_model_paths(model_size):
        if (path / "config.json").exists() and (path / "model.bin").exists():
            return path
    return None


def is_whisper_model_downloaded(model_size: str) -> bool:
    if resolve_local_model_path(model_size) is not None:
        return True

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False

    repo_id = f"Systran/faster-whisper-{model_size}"
    result = try_to_load_from_cache(
        repo_id,
        "config.json",
        cache_dir=_hf_cache_dir(),
    )
    return result is not None


def download_whisper_model(model_size: str) -> Path:
    if model_size not in SUPPORTED_WHISPER_MODELS:
        raise ValueError(f"Unsupported whisper model: {model_size}")

    local_path = resolve_local_model_path(model_size)
    if local_path is not None:
        return local_path

    from faster_whisper.utils import download_model  # noqa: PLC0415

    model_root = whisper_model_dir()
    model_root.mkdir(parents=True, exist_ok=True)
    cache_root = _hf_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    target_dir = _download_target_dir(model_size)
    target_dir.mkdir(parents=True, exist_ok=True)
    download_model(model_size, output_dir=str(target_dir), cache_dir=str(cache_root))
    _cleanup_download_artifacts(target_dir)

    local_path = resolve_local_model_path(model_size)
    if local_path is not None:
        return local_path

    return target_dir
