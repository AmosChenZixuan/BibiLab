"""Test demonstrating async DB layer issues - tests should fail with current sync implementation."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path):
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        with patch("bibilab.main.bibilab_home", return_value=tmp_path):
            with patch("bibilab.cleanup.bibilab_home", return_value=tmp_path):
                with patch("bibilab.routers.lists.bibilab_home", return_value=tmp_path):
                    with patch("bibilab.worker.bibilab_home", return_value=tmp_path):
                        with patch("bibilab.pipeline.transcribe.bibilab_home", return_value=tmp_path):
                            with patch("bibilab.pipeline.embed.bibilab_home", return_value=tmp_path):
                                with patch("pathlib.Path.home", return_value=tmp_path):
                                    yield tmp_path


@pytest.mark.asyncio
async def test_db_uses_aiosqlite_not_sqlite3(tmp_bibilab_home: Path):
    """
    Verify that get_db() uses aiosqlite for true async support.

    Current implementation uses sqlite3.connect() which is synchronous and
    blocks the event loop during I/O operations.

    This test checks that the connection returned is an aiosqlite connection,
    which has async-compatible execute() returning an async cursor.
    """
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()

    async with get_db() as db:
        import aiosqlite

        # The connection should be an instance of aiosqlite.Connection
        assert isinstance(db, aiosqlite.Connection), (
            f"Expected aiosqlite.Connection, got {type(db).__module__}.{type(db).__name__}. "
            "DB layer must use aiosqlite.connect() for true async support."
        )


@pytest.mark.asyncio
async def test_concurrent_db_operations_do_not_block_event_loop(tmp_bibilab_home: Path):
    """
    Test that concurrent DB operations don't block the event loop.

    With sync sqlite3, even though we use 'async with', the actual
    sqlite3.connect() and execute() calls block, preventing true
    concurrency.

    With aiosqlite, concurrent operations can run without blocking.
    """
    from bibilab.db import bootstrap_db, create_list, get_all_lists

    await bootstrap_db()

    # Create some data
    for i in range(10):
        await create_list(f"list-{i}", f"List {i}", "2026-01-01T00:00:00")

    # Verify the DB uses aiosqlite first
    import aiosqlite

    from bibilab.db import get_db

    async with get_db() as db:
        assert isinstance(db, aiosqlite.Connection), "get_db must return aiosqlite.Connection for true async support"

    # Now test concurrent operations work correctly
    async def concurrent_read():
        return await get_all_lists()

    # Run 20 concurrent reads
    results = await asyncio.gather(*[concurrent_read() for _ in range(20)])

    # All should return same data
    for r in results:
        assert len(r) == 10


@pytest.mark.asyncio
async def test_worker_never_double_dispatches_same_job(tmp_bibilab_home: Path):
    """
    Test that the worker loop's _in_flight set prevents double-dispatch
    even when multiple coroutines check the queue simultaneously.

    Bug scenario: if two coroutines in the worker loop both call
    get_pending_jobs() before either adds to _in_flight, they could
    both pick up the same job.

    The fix should atomically mark jobs as in-flight before releasing
    the lock that prevents other coroutines from picking them.
    """
    from bibilab.db import (
        JobStatus,
        bootstrap_db,
        create_job,
        get_pending_jobs,
    )

    await bootstrap_db()

    # Create several jobs
    job_ids = []
    for i in range(5):
        jid = await create_job("ingest", {"video_id": f"BV{i}", "list_id": "list-1"})
        job_ids.append(jid)

    # Simulate the worker's queue-picking logic
    in_flight: set[str] = set()
    concurrency = 2

    async def pick_jobs():
        nonlocal in_flight
        pending = await get_pending_jobs()
        queued = [j for j in pending if j["status"] == JobStatus.QUEUED.value and j["id"] not in in_flight]
        slots = concurrency - len(in_flight)
        picked = []
        for job in queued[:slots]:
            in_flight.add(job["id"])
            picked.append(job["id"])
        return picked

    # Run 10 concurrent pick attempts
    results = await asyncio.gather(*[pick_jobs() for _ in range(10)])

    # Flatten picked jobs
    all_picked = []
    for r in results:
        all_picked.extend(r)

    # Each job should only be picked once
    assert len(all_picked) == len(set(all_picked)), f"Double dispatch detected! Jobs picked: {all_picked}"

    # With only 5 jobs and concurrency=2, we should pick at most 2 jobs total
    # across all attempts (first two coroutines to pick)
    assert len(all_picked) <= 2, f"Should pick at most concurrency jobs, but picked {len(all_picked)}"
