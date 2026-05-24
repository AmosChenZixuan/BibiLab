import asyncio

import pytest

from bibilab.pipeline.chat_runs import (
    ChatRunRegistry,
    StreamBuffer,
    stream_from_buffer,
)


@pytest.mark.asyncio
async def test_buffer_replay_for_late_subscriber():
    buf = StreamBuffer(message_id="m1")
    buf.append({"type": "delta", "content": "a"})
    buf.append({"type": "delta", "content": "b"})

    received: list[dict] = []

    async def consume():
        async for ev in stream_from_buffer(buf):
            received.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    buf.append({"type": "delta", "content": "c"})
    buf.close("done")
    await task

    assert [e["content"] for e in received] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_buffer_fanout_to_multiple_subscribers():
    buf = StreamBuffer(message_id="m1")
    out_a, out_b = [], []

    async def consume(out):
        async for ev in stream_from_buffer(buf):
            out.append(ev)

    task_a = asyncio.create_task(consume(out_a))
    task_b = asyncio.create_task(consume(out_b))
    await asyncio.sleep(0)
    buf.append({"type": "delta", "content": "x"})
    buf.close("done")
    await asyncio.gather(task_a, task_b)

    assert out_a == out_b == [{"type": "delta", "content": "x"}]


@pytest.mark.asyncio
async def test_close_wakes_all_subscribers():
    buf = StreamBuffer(message_id="m1")

    async def consume():
        async for _ in stream_from_buffer(buf):
            pass

    tasks = [asyncio.create_task(consume()) for _ in range(3)]
    await asyncio.sleep(0)
    buf.close("done")
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)


@pytest.mark.asyncio
async def test_registry_cancel_idempotent():
    reg = ChatRunRegistry()

    async def long_task():
        await asyncio.sleep(0.01)

    t = asyncio.create_task(long_task())
    reg.register("m1", t)
    assert reg.cancel("m1") is True
    with pytest.raises(asyncio.CancelledError):
        await t
    assert reg.cancel("m1") is False  # already done


def test_registry_evict():
    reg = ChatRunRegistry()
    t = asyncio.get_event_loop().create_future()
    t.set_result(None)
    reg.register("m1", t)  # type: ignore[arg-type]
    assert reg.get("m1") is not None
    reg.evict("m1")
    assert reg.get("m1") is None
    reg.evict("m1")  # idempotent — does not raise
