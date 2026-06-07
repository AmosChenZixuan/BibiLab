"""Tests for GET /api/debug/messages/{message_id}."""

import json

import pytest

pytestmark = pytest.mark.integration


async def test_dump_endpoint_returns_200_with_content(client, tmp_bibilab_home):
    (tmp_bibilab_home / "debug").mkdir(parents=True)
    (tmp_bibilab_home / "debug" / "msg_xyz.json").write_text(json.dumps({"system": "s", "messages": []}))
    resp = await client.get("/debug/messages/msg_xyz")
    assert resp.status_code == 200
    assert resp.json() == {"system": "s", "messages": []}


async def test_dump_endpoint_returns_404_when_missing(client, tmp_bibilab_home):
    resp = await client.get("/debug/messages/msg_does_not_exist")
    assert resp.status_code == 404
    assert resp.json() == {"error": "dump_not_found", "message_id": "msg_does_not_exist"}
