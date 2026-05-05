from collections.abc import AsyncGenerator
from typing import Any


def an_async_generator(items: list[Any]) -> AsyncGenerator:
    async def gen():
        for item in items:
            yield item

    return gen()
