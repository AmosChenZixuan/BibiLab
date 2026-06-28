"""Tests for bibilab.model_registry."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.model_registry import (
    ModelSpec,
    _download_http_files,
    _target_dir,
    ensure,
    get_spec,
)


def _make_http_spec(target: Path) -> ModelSpec:
    return ModelSpec(
        id="test-http-spec",
        display_name="Test HTTP",
        kind="embedding",
        backend="http_files",
        size_mb=1,
        integrity_files=["a.txt", "sub/b.txt"],
        local_subdir="test-http",
        http_files=[
            ("http://example.invalid/a.txt", "a.txt"),
            ("http://example.invalid/sub/b.txt", "sub/b.txt"),
        ],
    )


class _FakeStreamResp:
    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, _chunk_size: int):
        yield b"hello"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_download_http_files_writes_integrity_files_and_renames_to_target(tmp_bibilab_home: Path):
    """Regression: prior `finally: shutil.rmtree(tmp)` wiped tmp before rename,
    making every http_files download raise 'atomic rename failed'.
    """
    target = tmp_bibilab_home / "models" / "test-http"
    spec = _make_http_spec(target)

    with patch("httpx.stream", return_value=_FakeStreamResp()):
        _download_http_files(spec, target)

    assert (target / "a.txt").read_bytes() == b"hello"
    assert (target / "sub" / "b.txt").read_bytes() == b"hello"


def test_ensure_raises_when_download_completes_but_integrity_fails(tmp_bibilab_home: Path):
    """Locks the post-download integrity verify added in 3af33e9."""
    spec = get_spec("multilingual-e5")

    def empty_download(_spec, target):
        target.mkdir(parents=True, exist_ok=True)

    with patch("bibilab.model_registry._download_http_files", side_effect=empty_download):
        with pytest.raises(RuntimeError, match="integrity check failed"):
            ensure(spec.id)


def test_modelspec_rejects_empty_integrity_files():
    """Empty list would make `_integrity_ok` vacuously True — guard at __post_init__."""
    with pytest.raises(ValueError, match="integrity_files"):
        ModelSpec(
            id="bad",
            display_name="Bad",
            kind="embedding",
            backend="http_files",
            size_mb=1,
            integrity_files=[],
            local_subdir="bad",
            http_files=[("http://example.invalid/x", "x")],
        )


def test_registry_sizes_corrected():
    """size_mb drives download UI/estimates; fp32 reranker (1060.9 MiB) and the
    e5 embedder (448.5 MiB) were declared far too low. The old fp32 280 actually
    matched the quantized file, not fp32."""
    assert get_spec("bge-reranker-base").size_mb == 1061
    assert get_spec("multilingual-e5").size_mb == 449


def test_quantized_reranker_spec_registered():
    """int8 quantized reranker (266.4 MiB) — registered as its own spec so the
    fp32 spec stays selectable; on-disk file normalized to model.onnx so the
    loader needs no filename branch."""
    spec = get_spec("bge-reranker-base-q")
    assert spec.kind == "reranker"
    assert spec.backend == "http_files"
    assert spec.size_mb == 266
    assert spec.integrity_files == ["model.onnx", "tokenizer.json"]
    # distinct dir from fp32 so a download never overwrites the other model
    assert spec.local_subdir != get_spec("bge-reranker-base").local_subdir
    assert spec.http_files is not None
    url_by_rel = {rel: url for url, rel in spec.http_files}
    assert url_by_rel["model.onnx"].endswith("onnx/model_quantized.onnx")
    assert "tokenizer.json" in url_by_rel


def test_ctpunc_spec_registered():
    spec = get_spec("ct-punc")
    assert spec.kind == "punctuation"
    assert spec.backend == "modelscope"
    assert spec.modelscope_id == "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
    assert spec.integrity_files == ["configuration.json"]
    assert spec.local_subdir == "asr/ct-punc"


def test_reranker_spec_id_constant_removed():
    """Selection now flows through config (cfg.rag.reranker_spec_id), so the old
    module constant is dead — importing it must fail to prevent a stale second
    source of truth."""
    import bibilab.model_registry as mr

    assert not hasattr(mr, "RERANKER_SPEC_ID")


def test_required_models_follows_configured_reranker_spec():
    """required_models must reflect the config-selected reranker, not a hardcoded
    spec — otherwise the download set diverges from what rerank.py loads."""
    from bibilab.config import BibilabConfig
    from bibilab.model_registry import required_models

    cfg = BibilabConfig()
    cfg.rag.reranker_spec_id = "bge-reranker-base"  # opt back into fp32
    ids = [s.id for s in required_models(cfg)]
    assert "bge-reranker-base" in ids
    assert "bge-reranker-base-q" not in ids


def test_ctpunc_is_required_unconditionally():
    from bibilab.config import BibilabConfig
    from bibilab.model_registry import PUNC_SPEC_ID, required_models

    cfg = BibilabConfig()
    ids = [s.id for s in required_models(cfg)]
    assert "ct-punc" in ids
    assert PUNC_SPEC_ID == "ct-punc"


def test_target_dir_routes_whisper_through_models_dir(tmp_bibilab_home: Path):
    """_target_dir must use the spec's local_subdir for whisper too (no special-case)."""
    spec = get_spec("large-v3")
    from bibilab.model_registry import _target_dir

    expected = _target_dir(spec)
    assert _target_dir(spec) == expected


def test_ensure_whisper_calls_load_model_with_download_root(tmp_bibilab_home: Path):
    """Bypass funasr's openai path: whisper.load_model(name, download_root=target) is
    the documented public API that writes the .pt to the caller's directory."""
    spec = get_spec("large-v3")
    expected_target = _target_dir(spec)

    def fake_load_model(name, download_root=None, **kwargs):
        assert name == "large-v3"
        # Mirror what openai-whisper does: write the .pt to <download_root>/<name>.pt
        Path(download_root).mkdir(parents=True, exist_ok=True)
        (Path(download_root) / f"{name}.pt").write_bytes(b"fake-checkpoint")
        # Return value is discarded by _download_whisper_warp
        return MagicMock()

    whisper_stub = MagicMock()
    whisper_stub.load_model = MagicMock(side_effect=fake_load_model)
    with patch.dict(sys.modules, {"whisper": whisper_stub}):
        mock = whisper_stub.load_model
        result = ensure(spec.id)

    assert result == expected_target
    assert (expected_target / "large-v3.pt").read_bytes() == b"fake-checkpoint"
    mock.assert_called_once()
    call = mock.call_args
    assert call.args[0] == "large-v3"
    assert call.kwargs.get("download_root") == str(expected_target)
