"""Tests for bibilab.pipeline.punctuate — char-offset alignment is model-free."""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.model_registry import get_spec
from bibilab.pipeline.chunk import _SENT_END
from bibilab.pipeline.punctuate import _align, _strip_punc
from bibilab.pipeline.transcribe import WhisperSegment


def _seg(text, start, end, speaker="SPK_0"):
    return WhisperSegment(start=start, end=end, text=text, speaker=speaker)


def test_strip_punc_removes_punctuation_and_spaces():
    assert _strip_punc("天花。板，明 显") == "天花板明显"


def test_align_invariant_violation_raises():
    # ct-punc must only INSERT; a rewritten char breaks the mapping.
    segs = [_seg("天花板", 0.0, 5.0)]
    with pytest.raises(ValueError):
        _align(segs, "天花版。")  # 板 -> 版 is a rewrite


def test_align_basic_single_speaker_two_sentences():
    # one VAD seg, ct-punc inserts a sentence break -> two sentences (G2: share span)
    segs = [_seg("天花板明显是地板", 10.0, 25.0)]
    out = _align(segs, "天花板。明显是地板。")
    assert [s.text for s in out] == ["天花板。", "明显是地板。"]
    # G2: both inherit the single VAD seg's span
    assert all(s.start == 10.0 and s.end == 25.0 for s in out)
    assert all(s.speaker == "SPK_0" for s in out)


def test_align_g1_sentence_spans_two_vad_segments():
    # ct-punc heals a false split across two VAD segs -> one sentence
    # G1: start = first seg.start, end = last seg.end
    segs = [_seg("天花", 10.0, 12.0), _seg("板明显", 12.0, 16.0)]
    out = _align(segs, "天花板明显。")
    assert [s.text for s in out] == ["天花板明显。"]
    assert out[0].start == 10.0 and out[0].end == 16.0


def test_align_g3_speaker_change_forces_cut_without_punctuation():
    # speaker switches mid-stream with no terminal punctuation -> two segments, first has no terminal
    segs = [_seg("你好", 0.0, 2.0, "SPK_0"), _seg("再见", 2.0, 4.0, "SPK_1")]
    out = _align(segs, "你好再见。")
    assert [(s.text, s.speaker) for s in out] == [("你好", "SPK_0"), ("再见。", "SPK_1")]
    assert not out[0].text.rstrip().endswith(_SENT_END)


def test_align_phantom_punctuation_only_segment_dissolves():
    # a VAD seg that is all punctuation contributes no chars -> not its own sentence
    segs = [_seg("你好。", 0.0, 2.0), _seg("。", 2.0, 3.0), _seg("再见", 3.0, 5.0)]
    out = _align(segs, "你好。再见。")
    assert [s.text for s in out] == ["你好。", "再见。"]


def test_align_trailing_buffer_flushed_without_terminal_punctuation():
    segs = [_seg("没有句号", 0.0, 3.0)]
    out = _align(segs, "没有句号")  # ct-punc left no terminal punctuation
    assert [s.text for s in out] == ["没有句号"]
    assert out[0].start == 0.0 and out[0].end == 3.0


def test_align_inserted_punctuation_outside_punc_set_is_kept():
    # ct-punc may emit marks outside _PUNC (quotes here). They must be treated as
    # inserted punctuation, not as a rewrite — the old fixed-set check raised here.
    segs = [_seg("他说你好", 0.0, 5.0)]
    out = _align(segs, '他说"你好"。')
    assert [s.text for s in out] == ['他说"你好"。']
    assert out[0].start == 0.0 and out[0].end == 5.0


# ---- punctuate() gate tests (mock _run_ctpunc) ----


def test_punctuate_non_zh_passthrough_no_model_call():
    from unittest.mock import patch

    from bibilab.pipeline.punctuate import punctuate

    segs = [_seg("hello world", 0.0, 3.0)]
    with patch("bibilab.pipeline.punctuate._run_ctpunc") as run:
        out = punctuate(segs, language="en")
    run.assert_not_called()
    assert out is segs


def test_punctuate_empty_segments_passthrough():
    from unittest.mock import patch

    from bibilab.pipeline.punctuate import punctuate

    with patch("bibilab.pipeline.punctuate._run_ctpunc") as run:
        out = punctuate([], language="zh")
    run.assert_not_called()
    assert out == []


def test_punctuate_zh_runs_ctpunc_and_aligns():
    from unittest.mock import patch

    from bibilab.pipeline.punctuate import punctuate

    segs = [_seg("天花板明显是地板", 10.0, 25.0)]
    with patch("bibilab.pipeline.punctuate._run_ctpunc", return_value="天花板。明显是地板。") as run:
        out = punctuate(segs, language="zh")
    run.assert_called_once_with("天花板明显是地板")
    assert [s.text for s in out] == ["天花板。", "明显是地板。"]


def test_punctuate_degrades_to_passthrough_on_alignment_failure(caplog):
    # A genuine rewrite (板 -> 版) breaks alignment. punctuate() must not crash
    # the pipeline — it degrades to the raw segments and logs a warning.
    import logging
    from unittest.mock import patch

    from bibilab.pipeline.punctuate import punctuate

    segs = [_seg("天花板", 0.0, 5.0)]
    with patch("bibilab.pipeline.punctuate._run_ctpunc", return_value="天花版。"):
        with caplog.at_level(logging.WARNING):
            out = punctuate(segs, language="zh")
    assert out is segs
    assert "ct-punc alignment failed" in caplog.text


# ---- _run_ctpunc construction (mocked sherpa_onnx) ----


def _reset_ctpunc_singleton() -> None:
    from bibilab.pipeline import punctuate as punctuate_mod

    punctuate_mod._ctpunc_model = None


def _stub_ensure(models_root: Path, spec_id: str) -> Path:
    return models_root / get_spec(spec_id).local_subdir


def test_run_ctpunc_builds_from_sherpa_ct_punc_spec_and_calls_add_punctuation(tmp_bibilab_home: Path):
    _reset_ctpunc_singleton()

    models_root = tmp_bibilab_home / "models"
    stub = MagicMock()
    fake_punct = MagicMock()
    fake_punct.add_punctuation.return_value = "天花板。"
    stub.OfflinePunctuation.return_value = fake_punct

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.punctuate.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.punctuate.interpreting_provider", return_value="cpu"),
    ):
        from bibilab.pipeline.punctuate import _run_ctpunc

        result = _run_ctpunc("天花板")

    assert result == "天花板。"
    fake_punct.add_punctuation.assert_called_once_with("天花板")

    model_kwargs = stub.OfflinePunctuationModelConfig.call_args.kwargs
    ct_punc_dir = models_root / get_spec("sherpa-ct-punc").local_subdir
    assert model_kwargs["ct_transformer"] == str(ct_punc_dir / "model.onnx")
    assert model_kwargs["provider"] == "cpu"


def test_run_ctpunc_reuses_singleton_across_calls(tmp_bibilab_home: Path):
    _reset_ctpunc_singleton()

    models_root = tmp_bibilab_home / "models"
    stub = MagicMock()
    fake_punct = MagicMock()
    fake_punct.add_punctuation.side_effect = lambda raw: raw + "。"
    stub.OfflinePunctuation.return_value = fake_punct

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.punctuate.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.punctuate.interpreting_provider", return_value="cpu"),
    ):
        from bibilab.pipeline.punctuate import _run_ctpunc

        _run_ctpunc("a")
        _run_ctpunc("b")

    assert stub.OfflinePunctuation.call_count == 1


class _HookedLock:
    """threading.Lock wrapper that runs a callback on every acquire — lets a
    test force a deterministic acquisition order across threads."""

    def __init__(self, on_acquire) -> None:
        self._lock = threading.Lock()
        self._on_acquire = on_acquire

    def acquire(self) -> bool:
        self._on_acquire()
        return self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def test_run_ctpunc_concurrent_cold_build_constructs_once(tmp_bibilab_home: Path):
    # Two threads racing the double-checked-locking build from cold: only one
    # construction, regardless of interleaving.
    _reset_ctpunc_singleton()

    models_root = tmp_bibilab_home / "models"
    stub = MagicMock()
    fake_punct = MagicMock()
    fake_punct.add_punctuation.side_effect = lambda raw: raw + "。"
    stub.OfflinePunctuation.return_value = fake_punct

    barrier = threading.Barrier(2)

    def call(raw: str) -> None:
        barrier.wait(timeout=5)
        from bibilab.pipeline.punctuate import _run_ctpunc

        _run_ctpunc(raw)

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.punctuate.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.punctuate.interpreting_provider", return_value="cpu"),
    ):
        threads = [threading.Thread(target=call, args=(f"raw{i}",)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

    assert stub.OfflinePunctuation.call_count == 1


def test_run_ctpunc_race_after_failure_does_not_leak_attributeerror(tmp_bibilab_home: Path):
    # Thread A builds the model then fails inference, resetting _ctpunc_model
    # to None. Thread B's outer unlocked "already built?" check must not go
    # stale — B is forced to check it while A still holds the model non-None,
    # then must re-check once it gets the lock (post A's reset) instead of
    # calling .add_punctuation() on None.
    _reset_ctpunc_singleton()

    models_root = tmp_bibilab_home / "models"
    stub = MagicMock()
    fake_punct = MagicMock()

    ready_for_b = threading.Event()  # A holds the lock, model built, not yet failed
    b_reached_lock = threading.Event()  # B has attempted to acquire the same lock

    call_count = {"n": 0}

    def add_punctuation_side_effect(raw: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            ready_for_b.set()
            assert b_reached_lock.wait(timeout=5), "B never reached the lock"
            raise RuntimeError("boom")
        return raw + "。"

    fake_punct.add_punctuation.side_effect = add_punctuation_side_effect
    stub.OfflinePunctuation.return_value = fake_punct

    results: dict[str, object] = {}

    def run_a() -> None:
        from bibilab.pipeline.punctuate import _run_ctpunc

        try:
            results["a"] = _run_ctpunc("raw_a")
        except Exception as exc:  # noqa: BLE001
            results["a"] = exc

    def run_b() -> None:
        from bibilab.pipeline.punctuate import _run_ctpunc

        assert ready_for_b.wait(timeout=5), "A never signalled it holds the built model"
        try:
            results["b"] = _run_ctpunc("raw_b")
        except Exception as exc:  # noqa: BLE001
            results["b"] = exc

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)

    def on_acquire() -> None:
        # Gating by thread identity (rather than an acquire-count) is necessary
        # because A may acquire and release the lock several times within a
        # single _run_ctpunc call — only B's first attempt matters here.
        if threading.current_thread() is thread_b:
            b_reached_lock.set()

    from bibilab.pipeline import punctuate as punctuate_mod

    with (
        patch.dict(sys.modules, {"sherpa_onnx": stub}),
        patch("bibilab.pipeline.punctuate.ensure", side_effect=lambda sid: _stub_ensure(models_root, sid)),
        patch("bibilab.pipeline.punctuate.interpreting_provider", return_value="cpu"),
        patch.object(punctuate_mod, "_ctpunc_lock", _HookedLock(on_acquire)),
    ):
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)

    assert isinstance(results["a"], RuntimeError)
    assert not isinstance(results["b"], AttributeError), (
        f"stale outer check let B call .add_punctuation() on a reset-to-None model: {results['b']!r}"
    )
    assert results["b"] == "raw_b。"
