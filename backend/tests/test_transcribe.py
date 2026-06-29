from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.config import TranscriptionConfig
from bibilab.model_registry import get_spec
from bibilab.pipeline.transcribe import (
    WhisperSegment,
    _resolve_device,
    build_speaker_namespace,
    format_turns,
)


def _seg(text, start=0.0, end=1.0, speaker="SPK_0"):
    return WhisperSegment(start=start, end=end, text=text, speaker=speaker)


def test_resolve_device_cpu_passthrough():
    assert _resolve_device(TranscriptionConfig(device="cpu")) == "cpu"


def test_resolve_device_cuda_when_gpu_present(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device(TranscriptionConfig(device="cuda")) == "cuda:0"


def test_resolve_device_clamps_cuda_to_cpu_without_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device(TranscriptionConfig(device="cuda")) == "cpu"


def test_format_turns_digest_variant_grouped_no_time_raw_label():
    segs = [
        _seg("你好。", 0.0, 2.0, "SPK_0"),
        _seg("今天天气不错。", 2.0, 5.0, "SPK_0"),
        _seg("是啊。", 5.0, 7.0, "SPK_1"),
    ]
    out = format_turns(segs)
    assert out == "[SPK_0] 你好。 今天天气不错。\n[SPK_1] 是啊。"


def test_format_turns_ui_variant_grouped_with_time_raw_label():
    segs = [_seg("你好。", 157.0, 160.0, "SPK_0")]
    out = format_turns(segs, include_time=True)
    assert out == "[SPK_0 @2:37] 你好。"


def test_format_turns_chat_variant_namespaced_with_time():
    segs = [
        _seg("你好。", 157.0, 160.0, "SPK_0"),
        _seg("再见。", 160.0, 162.0, "SPK_1"),
    ]
    ns = build_speaker_namespace(segs)  # SPK_0 -> 0, SPK_1 -> 1
    out = format_turns(segs, include_time=True, citation_index=3, speaker_namespace=ns)
    assert out == "[S3·SPK0 @2:37] 你好。\n[S3·SPK1 @2:40] 再见。"


def test_build_speaker_namespace_first_seen_order():
    segs = [_seg("a", 0, 1, "SPK_2"), _seg("b", 1, 2, "SPK_0"), _seg("c", 2, 3, "SPK_2")]
    assert build_speaker_namespace(segs) == {"SPK_2": 0, "SPK_0": 1}


def test_format_turns_none_speaker_renders_placeholder():
    out = format_turns([_seg("text", 0.0, 1.0, None)])
    assert out == "[SPK?] text"


def test_format_turns_time_shows_hours_only_past_an_hour():
    # < 1h: no hour field. >= 1h: H:MM:SS (3725s = 1:02:05).
    assert format_turns([_seg("a", 59.0, 60.0)], include_time=True) == "[SPK_0 @0:59] a"
    assert format_turns([_seg("b", 3725.0, 3726.0)], include_time=True) == "[SPK_0 @1:02:05] b"


@pytest.mark.asyncio
async def test_load_transcript_text_default_includes_time_grouped():
    from bibilab.pipeline.transcribe import load_transcript_text

    rows = [
        {"start_s": 0.0, "end_s": 2.0, "text": "你好。", "speaker": "SPK_0"},
        {"start_s": 2.0, "end_s": 4.0, "text": "再说一句。", "speaker": "SPK_0"},
    ]
    with patch("bibilab.db.get_transcript_segments", return_value=rows):
        out = await load_transcript_text("sid")
    assert out == "[SPK_0 @0:00] 你好。 再说一句。"  # grouped + time, raw label


@pytest.mark.asyncio
async def test_load_transcript_text_digest_variant_drops_time():
    from bibilab.pipeline.transcribe import load_transcript_text

    rows = [{"start_s": 9.0, "end_s": 11.0, "text": "重点。", "speaker": "SPK_0"}]
    with patch("bibilab.db.get_transcript_segments", return_value=rows):
        out = await load_transcript_text("sid", include_time=False)
    assert out == "[SPK_0] 重点。"


def _stub_ensure(models_root: Path, spec_id: str) -> Path:
    """Stub for model_registry.ensure(): return the dir matching the spec's local_subdir."""
    return models_root / get_spec(spec_id).local_subdir


def test_load_funasr_whisper_passes_local_checkpoint_as_model(tmp_bibilab_home: Path):
    """When cfg.model == 'large-v3', _load_funasr passes the pre-staged checkpoint
    path as `model` — funasr's openai branch local-loads an existing path (no
    download). Pins the wiring; the funasr contract itself is covered by the
    integration test below."""
    from bibilab.pipeline import transcribe as transcribe_mod

    # Reset module-level pipeline cache so _load_funasr actually builds
    transcribe_mod._funasr_pipeline = None
    transcribe_mod._funasr_key = None

    cfg = TranscriptionConfig(model="large-v3", device="cpu")
    mock_pipeline = MagicMock()

    models_root = tmp_bibilab_home / "models"
    funasr_stub = MagicMock()
    funasr_stub.AutoModel = MagicMock(return_value=mock_pipeline)
    with (
        patch.dict(sys.modules, {"funasr": funasr_stub}),
        patch(
            "bibilab.pipeline.transcribe.ensure",
            side_effect=lambda sid: _stub_ensure(models_root, sid),
        ),
    ):
        result = transcribe_mod._load_funasr(cfg)
        mock_auto = funasr_stub.AutoModel

    assert result is mock_pipeline
    assert mock_auto.called
    call_kwargs = mock_auto.call_args.kwargs
    assert call_kwargs["hub"] == "openai"
    assert call_kwargs["model"] == str(models_root / "asr" / "whisper" / "large-v3.pt")
    # Must NOT pass model_path: funasr's openai branch reads `model` and derives
    # model_path from it; a stray model_path with no `model` trips its assert.
    assert "model_path" not in call_kwargs


@pytest.mark.integration
def test_funasr_openai_local_checkpoint_resolves_without_download(tmp_path: Path):
    """Contract guard for the funasr boundary the unit test mocks away: passing an
    existing checkpoint as `model` with hub='openai' resolves to a local load
    (model='WhisperWarp', model_path=<path>) — no ~/.cache/whisper fetch. If this
    breaks, the production kwarg shape in _load_funasr is wrong."""
    from funasr.download.download_model_from_hub import download_model

    ckpt = tmp_path / "large-v3.pt"
    ckpt.write_bytes(b"")  # only the path is resolved here; contents irrelevant

    resolved = download_model(model=str(ckpt), hub="openai")

    assert resolved["model"] == "WhisperWarp"
    assert resolved["model_path"] == str(ckpt)
