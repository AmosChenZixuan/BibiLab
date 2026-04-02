import httpx
import pytest


@pytest.mark.asyncio
async def test_old_transcripts_endpoint_removed(client: httpx.AsyncClient):
    assert (await client.get("/transcripts/BV1abc")).status_code == 404
