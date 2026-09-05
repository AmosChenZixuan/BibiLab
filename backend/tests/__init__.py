import asyncio
import threading
from collections.abc import AsyncGenerator, Callable
from typing import Any


def an_async_generator(items: list[Any]) -> AsyncGenerator:
    async def gen():
        for item in items:
            yield item

    return gen()


def thread_signal() -> tuple[asyncio.Event, threading.Event, Callable[[], None]]:
    """(started, release, signal_started) for blocking a sync function run
    via asyncio.to_thread and cancelling it mid-flight from the test coroutine.

    `started` is an asyncio.Event, set via call_soon_threadsafe from the
    worker thread, so the test can `await started.wait()` instead of polling
    with a 10ms asyncio.sleep loop. `release` stays a plain threading.Event
    since the blocked function waits on it synchronously. `signal_started`
    is what the blocked function calls in place of a bare `started.set()`.

    Usage:
        started, release, signal_started = thread_signal()

        def _blocking_fn(*a, **k):
            signal_started()
            release.wait()
            return ...

        task = asyncio.create_task(worker._run_job(job))
        await started.wait()
        worker.cancel_job(job_id)
        release.set()
    """
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def signal_started() -> None:
        loop.call_soon_threadsafe(started.set)

    return started, release, signal_started


async def create_list(client, name: str) -> str:
    """Helper: POST /lists and return the new list id."""
    return (await client.post("/lists", json={"name": name})).json()["id"]
