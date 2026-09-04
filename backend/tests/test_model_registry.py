"""Tests for bibilab.model_registry."""

import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.model_registry import (
    ModelSpec,
    _download_http_archive,
    _download_http_files,
    _target_dir,
    ensure,
    get_spec,
    list_specs,
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
    def __init__(self, data: bytes = b"hello") -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, _chunk_size: int):
        yield self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_archive_spec() -> ModelSpec:
    return ModelSpec(
        id="test-archive-spec",
        display_name="Test Archive",
        kind="transcription",
        backend="http_archive",
        size_mb=1,
        integrity_files=["model.onnx"],
        local_subdir="test-archive",
        http_files=[("http://example.invalid/model.tar.bz2", "model.tar.bz2")],
    )


def _build_tar_bz2(tmp_path: Path, top_dir: str, files: dict[str, bytes]) -> bytes:
    """Build a real .tar.bz2 with a single top-level directory, mirroring how
    every k2-fsa release archive is shaped."""
    src = tmp_path / "src"
    (src / top_dir).mkdir(parents=True)
    for rel, content in files.items():
        path = src / top_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    archive_path = tmp_path / "built.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as tf:
        tf.add(src / top_dir, arcname=top_dir)
    return archive_path.read_bytes()


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


def test_download_http_archive_extracts_flattens_and_renames_to_target(tmp_path: Path, tmp_bibilab_home: Path):
    """AC2: a real single-top-dir .tar.bz2 (the shape of every k2-fsa release
    archive) extracts, flattens the wrapper dir away, and lands atomically."""
    archive_bytes = _build_tar_bz2(
        tmp_path,
        top_dir="sherpa-onnx-test-2024-01-01",
        files={"model.onnx": b"weights", "tokens.txt": b"a\nb\n"},
    )
    target = tmp_bibilab_home / "models" / "test-archive"
    spec = _make_archive_spec()

    with patch("httpx.stream", return_value=_FakeStreamResp(archive_bytes)):
        _download_http_archive(spec, target)

    assert (target / "model.onnx").read_bytes() == b"weights"
    assert (target / "tokens.txt").read_bytes() == b"a\nb\n"
    assert not (target.parent / f".{target.name}.partial").exists()


def test_download_http_archive_failure_leaves_no_partial_or_target(tmp_path: Path, tmp_bibilab_home: Path):
    """AC3: a broken archive (extraction raises) must not leave a .partial dir
    or a partially populated target — same guarantee _download_http_files gives."""
    target = tmp_bibilab_home / "models" / "test-archive"
    spec = _make_archive_spec()

    with patch("httpx.stream", return_value=_FakeStreamResp(b"not a real tar.bz2")):
        with pytest.raises(tarfile.ReadError):
            _download_http_archive(spec, target)

    assert not target.exists()
    assert not (target.parent / f".{target.name}.partial").exists()


def test_download_http_archive_requires_single_top_level_dir(tmp_path: Path, tmp_bibilab_home: Path):
    """Every known k2-fsa archive has exactly one top-level dir; a violation
    fails loud instead of silently nesting the wrong layout into target."""
    archive_path = tmp_path / "flat.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as tf:
        info = tarfile.TarInfo("loose_file.onnx")
        data = b"weights"
        info.size = len(data)
        import io

        tf.addfile(info, io.BytesIO(data))
    archive_bytes = archive_path.read_bytes()

    target = tmp_bibilab_home / "models" / "test-archive"
    spec = _make_archive_spec()

    with patch("httpx.stream", return_value=_FakeStreamResp(archive_bytes)):
        with pytest.raises(AssertionError):
            _download_http_archive(spec, target)

    assert not target.exists()
    assert not (target.parent / f".{target.name}.partial").exists()


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
    """size_mb drives download UI/estimates; the int8 reranker (266.4 MiB) and the
    e5 embedder (448.5 MiB) must round to their real on-disk sizes."""
    assert get_spec("bge-reranker-base-q").size_mb == 266
    assert get_spec("multilingual-e5").size_mb == 449


def test_quantized_reranker_is_sole_reranker_spec():
    """The int8 quantized reranker is the only reranker shipped; its on-disk file is
    normalized to model.onnx so the loader needs no filename branch, and it downloads
    the remote model_quantized.onnx. (size_mb is covered by test_registry_sizes_corrected.)"""
    rerankers = [s for s in list_specs() if s.kind == "reranker"]
    assert [s.id for s in rerankers] == ["bge-reranker-base-q"]

    spec = get_spec("bge-reranker-base-q")
    assert spec.backend == "http_files"
    assert spec.integrity_files == ["model.onnx", "tokenizer.json"]
    assert spec.http_files is not None
    url_by_rel = {rel: url for url, rel in spec.http_files}
    assert url_by_rel["model.onnx"].endswith("onnx/model_quantized.onnx")
    assert "tokenizer.json" in url_by_rel


def test_fp32_reranker_spec_removed():
    """The fp32 'bge-reranker-base' spec was deleted (one reranker ships).
    get_spec must fail loud rather than resolve a dead spec id."""
    with pytest.raises(ValueError):
        get_spec("bge-reranker-base")


def test_ctpunc_spec_registered():
    spec = get_spec("ct-punc")
    assert spec.kind == "punctuation"
    assert spec.backend == "modelscope"
    assert spec.modelscope_id == "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
    assert spec.integrity_files == ["configuration.json"]
    assert spec.local_subdir == "asr/ct-punc"


def test_reranker_spec_id_constant_is_quantized():
    """The reranker is a single module constant (mirrors EMBEDDING_SPEC_ID) instead
    of a config knob — it must name the registered int8 spec, the single source of
    truth rerank.py / required_models / health all resolve."""
    from bibilab.model_registry import RERANKER_SPEC_ID

    assert RERANKER_SPEC_ID == "bge-reranker-base-q"
    assert get_spec(RERANKER_SPEC_ID).kind == "reranker"


def test_required_models_includes_reranker_when_enabled():
    """required_models must list the one reranker spec (via the constant) when
    reranking is on, and omit it when off — so the download set matches what
    rerank.py loads, with no reference to the deleted config field."""
    from bibilab.config import BibilabConfig
    from bibilab.model_registry import RERANKER_SPEC_ID, required_models

    cfg = BibilabConfig()
    assert RERANKER_SPEC_ID in [s.id for s in required_models(cfg)]

    cfg.rag.reranking_enabled = False
    assert RERANKER_SPEC_ID not in [s.id for s in required_models(cfg)]


def test_ctpunc_is_required_unconditionally():
    """#685: transcribe.py/punctuate.py run on sherpa-onnx now, so the gate must
    require the sherpa ct-punc spec — not the now-unused FunASR one."""
    from bibilab.config import BibilabConfig
    from bibilab.model_registry import SHERPA_PUNC_SPEC_ID, required_models

    cfg = BibilabConfig()
    ids = [s.id for s in required_models(cfg)]
    assert SHERPA_PUNC_SPEC_ID in ids
    assert SHERPA_PUNC_SPEC_ID == "sherpa-ct-punc"


def test_resolve_transcription_spec_id_maps_public_model_names():
    """cfg.transcription.model values ("large-v3", "sensevoice-small") are stable
    public config strings; which concrete sherpa-onnx spec backs them is an
    implementation detail resolved here, in one place, shared by required_models()
    and transcribe.py's engine loader."""
    from bibilab.model_registry import (
        SHERPA_SENSEVOICE_SPEC_ID,
        SHERPA_WHISPER_SPEC_ID,
        resolve_transcription_spec_id,
    )

    assert resolve_transcription_spec_id("sensevoice-small") == SHERPA_SENSEVOICE_SPEC_ID
    assert resolve_transcription_spec_id("large-v3") == SHERPA_WHISPER_SPEC_ID


def test_required_models_uses_sherpa_vad_and_diarization_specs():
    from bibilab.config import BibilabConfig
    from bibilab.model_registry import SHERPA_DIARIZATION_SPEC_ID, SHERPA_VAD_SPEC_ID, required_models

    ids = [s.id for s in required_models(BibilabConfig())]
    assert SHERPA_VAD_SPEC_ID in ids
    assert SHERPA_DIARIZATION_SPEC_ID in ids


def test_required_models_transcription_model_resolves_to_sherpa_spec():
    from bibilab.config import BibilabConfig
    from bibilab.model_registry import SHERPA_SENSEVOICE_SPEC_ID, SHERPA_WHISPER_SPEC_ID, required_models

    cfg = BibilabConfig()
    cfg.transcription.model = "sensevoice-small"
    assert SHERPA_SENSEVOICE_SPEC_ID in [s.id for s in required_models(cfg)]

    cfg.transcription.model = "large-v3"
    assert SHERPA_WHISPER_SPEC_ID in [s.id for s in required_models(cfg)]


def test_ensure_dispatches_http_archive_backend(tmp_bibilab_home: Path):
    """AC1: ensure() must route an http_archive-backend spec to
    _download_http_archive, not silently fall through to another backend."""
    spec = get_spec("sherpa-ct-punc")
    assert spec.backend == "http_archive"

    def fake_download(_spec, target):
        target.mkdir(parents=True, exist_ok=True)
        for f in _spec.integrity_files:
            (target / f).write_bytes(b"x")

    with patch("bibilab.model_registry._download_http_archive", side_effect=fake_download) as mock:
        ensure(spec.id)
    mock.assert_called_once()


@pytest.mark.parametrize(
    ("spec_id", "backend", "kind"),
    [
        ("sherpa-sensevoice", "http_archive", "transcription"),
        ("sherpa-whisper-large-v3", "http_archive", "transcription"),
        ("sherpa-ct-punc", "http_archive", "punctuation"),
        ("sherpa-silero-vad", "http_files", "vad"),
        ("sherpa-campplus", "http_files", "diarization"),
    ],
)
def test_sherpa_specs_registered(spec_id: str, backend: str, kind: str):
    """AC4: the five new sherpa-onnx specs resolve with the right backend/kind,
    and land beside (not over) the existing PyTorch specs' local_subdir."""
    spec = get_spec(spec_id)
    assert spec.backend == backend
    assert spec.kind == kind
    assert spec.local_subdir.startswith("asr/")
    assert spec.http_files is not None


def test_http_archive_specs_have_exactly_one_url():
    """Structural guard: _download_http_archive assumes a single (url, name)
    tuple — a spec with zero or multiple would silently mis-flatten."""
    archive_specs = [s for s in list_specs() if s.backend == "http_archive"]
    assert archive_specs, "expected at least one http_archive spec"
    for spec in archive_specs:
        assert spec.http_files is not None
        assert len(spec.http_files) == 1


def test_existing_pytorch_specs_unchanged():
    """AC6 regression: the five pre-existing PyTorch specs still resolve
    with their original backend, untouched by the new sherpa specs."""
    assert get_spec("large-v3").backend == "whisper_warp"
    assert get_spec("sensevoice-small").backend == "modelscope"
    assert get_spec("cam++").backend == "modelscope"
    assert get_spec("fsmn-vad").backend == "modelscope"
    assert get_spec("ct-punc").backend == "modelscope"


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
