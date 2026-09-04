from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.config import TranscriptionConfig
from bibilab.model_registry import get_spec
from bibilab.pipeline.transcribe import (
    WhisperSegment,
    build_speaker_namespace,
    format_turns,
)


def _seg(text, start=0.0, end=1.0, speaker="SPK_0"):
    return WhisperSegment(start=start, end=end, text=text, speaker=speaker)


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


def _sherpa_stub() -> MagicMock:
    # MagicMock auto-vivifies every attribute access/call (so.VadModelConfig(),
    # .silero_vad.threshold = ..., .OfflineRecognizer.from_sense_voice(...).call_args,
    # etc.) — nothing needs explicit configuration for construction-only assertions.
    return MagicMock()


def _reset_sherpa_engine_cache() -> None:
    from bibilab.pipeline import transcribe as transcribe_mod

    transcribe_mod._sherpa_engine = None
    transcribe_mod._sherpa_engine_key = None


def test_load_sherpa_sensevoice_resolves_spec_paths_and_provider(tmp_bibilab_home: Path):
    """cfg.model='sensevoice-small' must build the recognizer from the sherpa
    SenseVoice spec's own paths and pass the shared provider policy through —
    not a hardcoded path or a second provider-selection mechanism."""
    _reset_sherpa_engine_cache()
    from bibilab.pipeline import transcribe as transcribe_mod

    cfg = TranscriptionConfig(model="sensevoice-small", language="auto")
    models_root = tmp_bibilab_home / "models"
    stub = _sherpa_stub()

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.transcribe.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.transcribe.interpreting_provider", return_value="cpu"),
    ):
        transcribe_mod._load_sherpa(cfg)

    kwargs = stub.OfflineRecognizer.from_sense_voice.call_args.kwargs
    sensevoice_dir = models_root / get_spec("sherpa-sensevoice").local_subdir
    assert kwargs["model"] == str(sensevoice_dir / "model.int8.onnx")
    assert kwargs["tokens"] == str(sensevoice_dir / "tokens.txt")
    assert kwargs["provider"] == "cpu"
    assert kwargs["language"] == "auto"
    stub.OfflineRecognizer.from_whisper.assert_not_called()


def test_load_sherpa_whisper_forces_english_regardless_of_cfg_language(tmp_bibilab_home: Path):
    """cfg.model='large-v3' must build from the sherpa Whisper spec and always
    construct with language='en' — English is the only validated case; any
    other language is untested and known-bad (0.40 CER on Chinese)."""
    _reset_sherpa_engine_cache()
    from bibilab.pipeline import transcribe as transcribe_mod

    cfg = TranscriptionConfig(model="large-v3", language="zh")  # deliberately mismatched
    models_root = tmp_bibilab_home / "models"
    stub = _sherpa_stub()

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.transcribe.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.transcribe.interpreting_provider", return_value="cpu"),
    ):
        transcribe_mod._load_sherpa(cfg)

    kwargs = stub.OfflineRecognizer.from_whisper.call_args.kwargs
    whisper_dir = models_root / get_spec("sherpa-whisper-large-v3").local_subdir
    assert kwargs["encoder"] == str(whisper_dir / "large-v3-encoder.int8.onnx")
    assert kwargs["decoder"] == str(whisper_dir / "large-v3-decoder.int8.onnx")
    assert kwargs["tokens"] == str(whisper_dir / "large-v3-tokens.txt")
    assert kwargs["language"] == "en"
    assert kwargs["task"] == "transcribe"
    stub.OfflineRecognizer.from_sense_voice.assert_not_called()


def test_load_sherpa_vad_uses_measured_defaults(tmp_bibilab_home: Path):
    """VAD threshold=0.3, min_silence_duration=0.25 — a measured config that beats
    sherpa's own defaults (0.5 / 0.50) on CER and speaker agreement."""
    _reset_sherpa_engine_cache()
    from bibilab.pipeline import transcribe as transcribe_mod

    cfg = TranscriptionConfig(model="sensevoice-small")
    models_root = tmp_bibilab_home / "models"
    stub = _sherpa_stub()

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.transcribe.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.transcribe.interpreting_provider", return_value="cpu"),
    ):
        engine = transcribe_mod._load_sherpa(cfg)

    assert engine.vad_cfg.silero_vad.threshold == 0.3
    assert engine.vad_cfg.silero_vad.min_silence_duration == 0.25
    assert engine.vad_cfg.silero_vad.max_speech_duration == 15.0
    assert engine.vad_cfg.provider == "cpu"


def test_load_sherpa_speaker_extractor_resolves_spec_path_and_provider(tmp_bibilab_home: Path):
    _reset_sherpa_engine_cache()
    from bibilab.pipeline import transcribe as transcribe_mod

    cfg = TranscriptionConfig(model="sensevoice-small")
    models_root = tmp_bibilab_home / "models"
    stub = _sherpa_stub()

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.transcribe.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.transcribe.interpreting_provider", return_value="cpu"),
    ):
        transcribe_mod._load_sherpa(cfg)

    kwargs = stub.SpeakerEmbeddingExtractorConfig.call_args.kwargs
    spk_dir = models_root / get_spec("sherpa-campplus").local_subdir
    assert kwargs["model"] == str(spk_dir / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
    assert kwargs["provider"] == "cpu"


def test_load_sherpa_caches_by_model_and_language(tmp_bibilab_home: Path):
    """The engine singleton must rebuild when either cfg.model or cfg.language
    changes (both are real construction-time inputs to the recognizer), and
    reuse the cached engine when neither changes."""
    _reset_sherpa_engine_cache()
    from bibilab.pipeline import transcribe as transcribe_mod

    models_root = tmp_bibilab_home / "models"
    stub = _sherpa_stub()

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.transcribe.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.transcribe.interpreting_provider", return_value="cpu"),
    ):
        e1 = transcribe_mod._load_sherpa(TranscriptionConfig(model="sensevoice-small", language="auto"))
        e2 = transcribe_mod._load_sherpa(TranscriptionConfig(model="sensevoice-small", language="auto"))
        e3 = transcribe_mod._load_sherpa(TranscriptionConfig(model="sensevoice-small", language="zh"))

    assert e1 is e2
    assert e1 is not e3
    assert stub.OfflineRecognizer.from_sense_voice.call_count == 2


def test_load_sherpa_builds_singleton_exactly_once_under_concurrent_entry(tmp_bibilab_home: Path):
    """Two threads racing to build the sherpa singleton for the same cfg must
    construct it exactly once. Thread A is deliberately held mid-build (via an
    Event, patched into interpreting_provider — called right after the cache
    check, inside the lock) while thread B signals the instant it starts
    attempting entry, so the main thread releases A only once B is genuinely
    contending — no sleep, so this can't false-pass under scheduler load."""
    import threading

    _reset_sherpa_engine_cache()
    from bibilab.pipeline import transcribe as transcribe_mod

    cfg = TranscriptionConfig(model="sensevoice-small", language="auto")
    models_root = tmp_bibilab_home / "models"
    stub = _sherpa_stub()

    build_started = threading.Event()
    release_build = threading.Event()
    b_entered = threading.Event()

    def blocking_provider() -> str:
        build_started.set()
        release_build.wait(timeout=2)
        return "cpu"

    def run_b() -> None:
        b_entered.set()
        results.append(transcribe_mod._load_sherpa(cfg))

    results: list = []
    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.transcribe.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.transcribe.interpreting_provider", side_effect=blocking_provider),
    ):
        thread_a = threading.Thread(target=lambda: results.append(transcribe_mod._load_sherpa(cfg)))
        thread_a.start()
        assert build_started.wait(timeout=2), "thread A never reached construction"

        thread_b = threading.Thread(target=run_b)
        thread_b.start()
        assert b_entered.wait(timeout=2), "thread B never started"
        release_build.set()

        thread_a.join(timeout=2)
        thread_b.join(timeout=2)

    assert stub.OfflineRecognizer.from_sense_voice.call_count == 1
    assert results[0] is results[1]


def _fake_engine(recognized: list[tuple[str, str | None]], embeddings: list[list[float]]):
    """A fake _SherpaEngine whose recognizer/spk_extractor streams return the given
    per-span (text, lang) pairs and embeddings, in call order."""
    from bibilab.pipeline.transcribe import _SherpaEngine

    recognizer = MagicMock()
    texts_iter = iter(recognized)

    def make_asr_stream():
        stream = MagicMock()
        text, lang = next(texts_iter)
        stream.result.text = text
        stream.result.lang = lang
        return stream

    recognizer.create_stream.side_effect = make_asr_stream

    spk = MagicMock()
    emb_iter = iter(embeddings)
    spk.create_stream.side_effect = lambda: MagicMock()
    spk.compute.side_effect = lambda _stream: emb_iter.__next__()

    return _SherpaEngine(vad_cfg=MagicMock(), recognizer=recognizer, spk_extractor=spk)


def _patch_transcribe_sherpa(engine, spans):
    return (
        patch("bibilab.pipeline.transcribe._load_sherpa", return_value=engine),
        patch("bibilab.pipeline.transcribe._vad_spans", return_value=spans),
        patch("soundfile.read", return_value=([0.0], 16000)),
    )


def test_transcribe_sherpa_does_not_hold_lock_during_inference(tmp_path: Path):
    """_transcribe_lock guards singleton construction only — decode must run
    with the lock free so concurrent callers actually overlap, matching the
    measured throughput gain from a shared model."""
    from bibilab.pipeline import transcribe as transcribe_mod
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(recognized=[("你好", None)], embeddings=[[1.0, 0.0]])
    lock_states_during_decode = []
    real_create_stream = engine.recognizer.create_stream

    def spying_create_stream():
        lock_states_during_decode.append(transcribe_mod._transcribe_lock.locked())
        return real_create_stream()

    engine.recognizer.create_stream = spying_create_stream

    spans = [(0.0, 1.0)]
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans)
    with p1, p2, p3:
        _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="sensevoice-small"))

    assert lock_states_during_decode == [False]


def test_cluster_speakers_splits_dissimilar_and_merges_similar_embeddings():
    from bibilab.pipeline.transcribe import _cluster_speakers

    assert _cluster_speakers([]) == []
    assert _cluster_speakers([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) == [0, 1]  # orthogonal
    assert _cluster_speakers([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) == [0, 0]  # identical


def test_transcribe_sherpa_assembles_segments_and_clusters_distinct_speakers(tmp_path: Path):
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(
        recognized=[("你好", None), ("再见", None)],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # orthogonal -> distinct clusters
    )
    spans = [(0.0, 1.0), (1.0, 2.0)]
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans)
    with p1, p2, p3:
        segments, _ = _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="sensevoice-small"))

    assert [(s.text, s.start, s.end) for s in segments] == [("你好", 0.0, 1.0), ("再见", 1.0, 2.0)]
    assert segments[0].speaker != segments[1].speaker


def test_transcribe_sherpa_clusters_similar_embeddings_as_one_speaker(tmp_path: Path):
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(
        recognized=[("你好", None), ("再见", None)],
        embeddings=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # identical -> same cluster
    )
    spans = [(0.0, 1.0), (1.0, 2.0)]
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans)
    with p1, p2, p3:
        segments, _ = _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="sensevoice-small"))

    assert segments[0].speaker == segments[1].speaker


def test_transcribe_sherpa_skips_empty_text_spans(tmp_path: Path):
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(
        recognized=[("", None), ("你好", None)],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    spans = [(0.0, 1.0), (1.0, 2.0)]
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans)
    with p1, p2, p3:
        segments, _ = _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="sensevoice-small"))

    assert [s.text for s in segments] == ["你好"]


def test_transcribe_sherpa_no_segments_returns_empty_and_none(tmp_path: Path):
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(recognized=[], embeddings=[])
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans=[])
    with p1, p2, p3:
        segments, lang = _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="sensevoice-small"))

    assert segments == []
    assert lang is None


def test_transcribe_sherpa_detected_language_explicit_cfg_wins(tmp_path: Path):
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(recognized=[("你好", "en")], embeddings=[[1.0, 0.0]])
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans=[(0.0, 1.0)])
    with p1, p2, p3:
        _, lang = _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="sensevoice-small", language="zh"))

    assert lang == "zh"


def test_transcribe_sherpa_whisper_detected_language_forced_english(tmp_path: Path):
    """large-v3 is always constructed with language='en' (see _load_sherpa); the
    returned detected_language must match, regardless of what the (untested for
    non-English audio) stream result reports."""
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(recognized=[("hello", None)], embeddings=[[1.0, 0.0]])
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans=[(0.0, 1.0)])
    with p1, p2, p3:
        _, lang = _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="large-v3", language="auto"))

    assert lang == "en"


def test_transcribe_sherpa_whisper_forces_english_even_with_mismatched_explicit_cfg_language(tmp_path: Path):
    """The model-forced language must win over an explicit-but-wrong cfg.language —
    large-v3 always decodes English regardless of cfg.language (see _load_sherpa), so
    reporting cfg.language='zh' here would send genuinely English text into zh ct-punc
    downstream and produce garbled punctuation."""
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(recognized=[("hello", None)], embeddings=[[1.0, 0.0]])
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans=[(0.0, 1.0)])
    with p1, p2, p3:
        _, lang = _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="large-v3", language="zh"))

    assert lang == "en"


def test_transcribe_sherpa_sensevoice_auto_uses_first_seen_recognizer_language(tmp_path: Path):
    from bibilab.pipeline.transcribe import _transcribe_sherpa

    engine = _fake_engine(
        recognized=[("你好", "zh"), ("hi", "en")],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    spans = [(0.0, 1.0), (1.0, 2.0)]
    p1, p2, p3 = _patch_transcribe_sherpa(engine, spans)
    with p1, p2, p3:
        _, lang = _transcribe_sherpa(tmp_path / "a.wav", TranscriptionConfig(model="sensevoice-small", language="auto"))

    assert lang == "zh"  # first segment's reported language


class _FakeVadSegment:
    def __init__(self, start: int, samples: list[float]) -> None:
        self.start = start
        self.samples = samples


class _FakeVad:
    """Emits nothing mid-stream, then one trailing span at flush() — the common
    case for a short clip with no silence gap before the audio ends."""

    def __init__(self, _config, buffer_size_in_seconds=100) -> None:
        self._flushed = False
        self._popped = False

    def accept_waveform(self, _chunk) -> None:
        pass

    def empty(self) -> bool:
        return not (self._flushed and not self._popped)

    @property
    def front(self) -> _FakeVadSegment:
        return _FakeVadSegment(start=1600, samples=[0.0] * 3200)

    def pop(self) -> None:
        self._popped = True

    def flush(self) -> None:
        self._flushed = True


def test_vad_spans_flushes_trailing_span_at_end_of_audio():
    from bibilab.pipeline.transcribe import _vad_spans

    fake_so = MagicMock()
    fake_so.VoiceActivityDetector = _FakeVad
    vad_cfg = MagicMock()
    vad_cfg.silero_vad.window_size = 512

    with patch.dict(sys.modules, {"sherpa_onnx": fake_so}):
        spans = _vad_spans(vad_cfg, [0.0] * 2000, 16000)

    assert spans == [(1600 / 16000, (1600 + 3200) / 16000)]


class _FakeVadRecorder:
    """Records the total sample count it's ever fed via accept_waveform — used to
    catch the windowing loop silently dropping a trailing (or, for very short
    clips, the entire) chunk."""

    def __init__(self, _config, buffer_size_in_seconds=100) -> None:
        self.fed = 0

    def accept_waveform(self, chunk) -> None:
        self.fed += len(chunk)

    def empty(self) -> bool:
        return True

    def pop(self) -> None:
        pass

    def flush(self) -> None:
        pass


def test_vad_spans_feeds_every_sample_including_a_trailing_partial_window():
    from bibilab.pipeline.transcribe import _vad_spans

    recorder = _FakeVadRecorder(None)
    fake_so = MagicMock()
    fake_so.VoiceActivityDetector = MagicMock(return_value=recorder)
    vad_cfg = MagicMock()
    vad_cfg.silero_vad.window_size = 512

    # 1536 = exactly 3 windows; +100 = one more short trailing chunk. Neither the
    # exact multiple's last window nor the trailing partial window may be dropped.
    samples = [0.0] * 1636
    with patch.dict(sys.modules, {"sherpa_onnx": fake_so}):
        _vad_spans(vad_cfg, samples, 16000)

    assert recorder.fed == len(samples)


def test_vad_spans_feeds_a_clip_shorter_than_one_window():
    from bibilab.pipeline.transcribe import _vad_spans

    recorder = _FakeVadRecorder(None)
    fake_so = MagicMock()
    fake_so.VoiceActivityDetector = MagicMock(return_value=recorder)
    vad_cfg = MagicMock()
    vad_cfg.silero_vad.window_size = 512

    samples = [0.0] * 200  # shorter than one window
    with patch.dict(sys.modules, {"sherpa_onnx": fake_so}):
        _vad_spans(vad_cfg, samples, 16000)

    assert recorder.fed == len(samples)


def test_transcribe_unknown_model_raises_pipeline_error():
    from bibilab.pipeline.audio import PipelineError
    from bibilab.pipeline.transcribe import transcribe

    with pytest.raises(PipelineError):
        transcribe(Path("/tmp/does-not-matter.wav"), TranscriptionConfig(model="not-a-real-model"))


@pytest.mark.parametrize(
    "module",
    ["bibilab.pipeline.transcribe", "bibilab.pipeline.punctuate"],
)
def test_no_torch_or_funasr_import_remains(module: str):
    """No torch or funasr import remains on the transcription/punctuation path."""
    import importlib

    source = Path(importlib.import_module(module).__file__).read_text()
    for banned in ("import torch", "from torch", "import funasr", "from funasr"):
        assert banned not in source, f"{module} still contains {banned!r}"
