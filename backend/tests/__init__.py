from collections.abc import AsyncGenerator
from typing import Any


def an_async_generator(items: list[Any]) -> AsyncGenerator:
    async def gen():
        for item in items:
            yield item

    return gen()


async def create_list(client, name: str) -> str:
    """Helper: POST /lists and return the new list id."""
    return (await client.post("/lists", json={"name": name})).json()["id"]
