import pytest

from bibilab.pipeline.transcribe import WhisperSegment, format_transcript_text


def test_format_transcript_text_matches_legacy_line_format():
    segs = [
        WhisperSegment(start=0.0, end=2.0, text="你好。", speaker="SPK_0"),
        WhisperSegment(start=3725.0, end=3726.0, text="结束。", speaker=None),
    ]
    out = format_transcript_text(segs)
    assert out == "[00:00:00] 你好。 [SPK_0]\n[01:02:05] 结束。"


def test_format_transcript_text_empty():
    assert format_transcript_text([]) == ""


@pytest.mark.asyncio
async def test_load_transcript_text_roundtrip(tmp_bibilab_home):
    from bibilab.db import _now, bootstrap_db, create_list, get_db, write_transcript_segments
    from bibilab.pipeline.transcribe import load_transcript_text

    await bootstrap_db()
    await create_list("list-1", "L", _now())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id) VALUES (?, ?, ?, ?)",
            ("src-1", "BV1", "bilibili", "list-1"),
        )
        await db.commit()
    await write_transcript_segments("src-1", [WhisperSegment(start=0.0, end=2.0, text="你好。", speaker="SPK_0")])
    assert await load_transcript_text("src-1") == "[00:00:00] 你好。 [SPK_0]"
