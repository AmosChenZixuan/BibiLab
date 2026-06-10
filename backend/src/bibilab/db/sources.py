"""Source table CRUD + atomic source+segments+sections writer."""

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from bibilab.db.connection import _now, get_db


def _in_placeholders(ids: list[str]) -> str:
    return ",".join("?" * len(ids)) if ids else ""


async def _exec_write_source(
    db: aiosqlite.Connection,
    *,
    source_id: str,
    video_id: str,
    platform: str,
    list_id: str,
    title: str,
    cover_url: str | None,
    source_url: str,
    duration_seconds: int,
    uploader: str,
    language: str | None,
    whisper_model: str,
    ai_model: str,
    settings_snapshot: dict[str, Any],
    series_name: str | None = None,
    sequence_number: int | None = None,
    season_number: int | None = None,
) -> None:
    """Upsert a source row on the given connection. Does NOT commit — the caller
    owns the transaction (lets a source + its segments commit atomically)."""
    cursor = await db.execute("SELECT id FROM sources WHERE video_id = ? AND list_id = ?", (video_id, list_id))
    existing = await cursor.fetchone()

    await db.execute(
        """
        INSERT INTO sources
            (id, video_id, platform, list_id, title,
             cover_url, source_url, duration_seconds, uploader,
             language, whisper_model, ai_model,
             processed_at, settings_snapshot,
             series_name, sequence_number, season_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id, list_id) DO UPDATE SET
            platform=excluded.platform,
            title=excluded.title,
            cover_url=excluded.cover_url,
            source_url=excluded.source_url,
            duration_seconds=excluded.duration_seconds,
            uploader=excluded.uploader,
            language=excluded.language,
            whisper_model=excluded.whisper_model,
            ai_model=excluded.ai_model,
            processed_at=excluded.processed_at,
            settings_snapshot=excluded.settings_snapshot,
            series_name=COALESCE(excluded.series_name, series_name),
            sequence_number=COALESCE(excluded.sequence_number, sequence_number),
            season_number=COALESCE(excluded.season_number, season_number)
        """,
        (
            source_id,
            video_id,
            platform,
            list_id,
            title,
            cover_url,
            source_url,
            duration_seconds,
            uploader,
            language,
            whisper_model,
            ai_model,
            _now(),
            json.dumps(settings_snapshot),
            series_name,
            sequence_number,
            season_number,
        ),
    )

    if existing is None:
        cursor = await db.execute("SELECT thumbnail_source_id FROM lists WHERE id = ?", (list_id,))
        list_row = await cursor.fetchone()
        if list_row is not None and list_row["thumbnail_source_id"] is None:
            await db.execute(
                "UPDATE lists SET thumbnail_source_id = ? WHERE id = ?",
                (source_id, list_id),
            )


async def _exec_write_transcript_segments(db: aiosqlite.Connection, source_id: str, segments: list) -> None:
    """Replace a source's transcript segments on the given connection. Does NOT
    commit. `segments` is a list of WhisperSegment (start, end, text, speaker)."""
    await db.execute("DELETE FROM transcript_segments WHERE source_id = ?", (source_id,))
    await db.executemany(
        "INSERT INTO transcript_segments (source_id, seq, start_s, end_s, speaker, text) VALUES (?, ?, ?, ?, ?, ?)",
        [(source_id, i, s.start, s.end, s.speaker, s.text) for i, s in enumerate(segments)],
    )


async def _exec_write_sections(
    db: aiosqlite.Connection,
    source_id: str,
    sections: list,
    section_digests: list,
) -> None:
    """Replace a source's section rows on the given connection. Does NOT
    commit. Mirrors `_exec_write_transcript_segments` (DELETE+INSERT) so
    re-ingest replaces cleanly. Rows have a surrogate `id`; re-ingest changes it.
    """
    if len(section_digests) != len(sections):
        raise ValueError(
            f"_exec_write_sections: section_digests length {len(section_digests)} != sections length {len(sections)}"
        )
    await db.execute("DELETE FROM sections WHERE source_id = ?", (source_id,))
    rows = [
        (
            source_id,
            i,
            s.seg_start,
            s.seg_end,
            s.token_count,
            s.timestamp_start,
            s.timestamp_end,
            sd.summary,
            json.dumps(sd.keywords),
        )
        for i, (s, sd) in enumerate(zip(sections, section_digests))
    ]
    await db.executemany(
        "INSERT INTO sections (source_id, seq, seg_start, seg_end, "
        "token_count, timestamp_start, timestamp_end, summary, keywords) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


async def write_source_with_segments(
    *,
    segments: list,
    sections: list | None = None,
    section_digests: list | None = None,
    **source_fields: Any,
) -> None:
    """Atomically upsert a source row, its transcript segments, and its
    derived section rows in one transaction. Either all three land or
    none — no orphaned source row, no compensating delete, no orphan
    section rows.
    """
    if sections is not None and section_digests is None:
        raise ValueError("write_source_with_segments: section_digests is required when sections is provided")
    async with get_db() as db:
        await _exec_write_source(db, **source_fields)
        await _exec_write_transcript_segments(db, source_fields["source_id"], segments)
        if sections is not None:
            await _exec_write_sections(db, source_fields["source_id"], sections, section_digests)
        await db.commit()


async def get_source(source_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM sources WHERE id=?", (source_id,))
        return await cursor.fetchone()


async def write_transcript_segments(source_id: str, segments: list) -> None:
    """Replace all transcript segments for a source. `segments` is a list of
    WhisperSegment (start, end, text, speaker). Idempotent (DELETE then INSERT).

    Standalone segment-write primitive. Production writes segments atomically
    with the source via `write_source_with_segments`; this is the unit-level
    seam the segments-table tests exercise as their subject (round-trip,
    cascade-on-delete, FK orphan rejection) — keep it even with zero prod
    callers, those tests can't express the orphan case via the atomic path."""
    async with get_db() as db:
        await _exec_write_transcript_segments(db, source_id, segments)
        await db.commit()


async def get_sources_for_list(list_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM sources WHERE list_id=? ORDER BY processed_at ASC",
            (list_id,),
        )
        return await cursor.fetchall()


async def get_source_facets(source_ids: list[str]) -> dict[str, dict[str, int | None]]:
    """Return {source_id: {"sequence_number": int|None, "season_number": int|None}}.

    Only sources found in the table are included. Empty input → {}.
    series_name is intentionally not returned — fuzzy string match is deferred
    until we see real-world use.
    """
    if not source_ids:
        return {}
    async with get_db() as db:
        placeholders = _in_placeholders(source_ids)
        cursor = await db.execute(
            f"SELECT id, sequence_number, season_number FROM sources WHERE id IN ({placeholders})",
            source_ids,
        )
        rows = await cursor.fetchall()
        return {
            row["id"]: {
                "sequence_number": row["sequence_number"],
                "season_number": row["season_number"],
            }
            for row in rows
        }


async def delete_source(source_id: str) -> None:
    async with get_db() as db:
        await db.execute("UPDATE lists SET thumbnail_source_id = NULL WHERE thumbnail_source_id = ?", (source_id,))
        await db.execute("DELETE FROM sources WHERE id=?", (source_id,))
        await db.commit()


async def delete_sources_for_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE lists SET thumbnail_source_id = NULL WHERE id = ? AND thumbnail_source_id IN "
            "(SELECT id FROM sources WHERE list_id = ?)",
            (list_id, list_id),
        )
        await db.execute("DELETE FROM sources WHERE list_id=?", (list_id,))
        await db.commit()
