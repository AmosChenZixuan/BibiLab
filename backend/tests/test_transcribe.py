from unittest.mock import patch

import pytest

from bibilab.pipeline.transcribe import WhisperSegment, build_speaker_namespace, format_turns


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
