from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_get_artifacts_empty(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    resp = await client.get(f"/lists/{list_id}/artifacts")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_artifacts_not_found(client: httpx.AsyncClient):
    resp = await client.get("/lists/nonexistent-list/artifacts")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_artifact_queues_job(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    resp = await client.post(
        f"/lists/{list_id}/artifacts",
        json={
            "type": "summary",
            "prompt": "Summarize the videos",
            "source_ids": ["src1", "src2"],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    # POST now returns job info, not artifact info
    assert data["type"] == "artifact"
    assert data["status"] == "queued"
    assert data["progress"] == 0
    assert data["error"] is None
    assert "id" in data  # job_id
    assert "created_at" in data
    assert "updated_at" in data
    # meta contains artifact info
    assert data["meta"]["list_id"] == list_id
    assert data["meta"]["type"] == "summary"
    assert data["meta"]["prompt"] == "Summarize the videos"
    assert data["meta"]["source_ids"] == ["src1", "src2"]
    assert "artifact_id" in data["meta"]


@pytest.mark.asyncio
async def test_create_artifact_not_found_list(client: httpx.AsyncClient):
    resp = await client.post(
        "/lists/nonexistent/artifacts",
        json={
            "type": "summary",
            "prompt": "Summarize",
            "source_ids": [],
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_artifact_metadata(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]

    # Create artifact via direct DB insert (simulating worker creating it)
    artifact_id = "art-001"
    content_path = f"artifacts/{list_id}/{artifact_id}.md"
    artifact_file = tmp_bibilab_home / content_path
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text("# Test Artifact\n\nContent here.", encoding="utf-8")

    from bibilab.db import create_artifact

    await create_artifact(
        artifact_id=artifact_id,
        list_id=list_id,
        name="Test Artifact",
        type="summary",
        prompt="Summarize videos",
        source_ids=["src1", "src2"],
        status="completed",
        content_path=content_path,
    )

    resp = await client.get(f"/artifacts/{artifact_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == artifact_id
    assert data["list_id"] == list_id
    assert data["name"] == "Test Artifact"
    assert data["type"] == "summary"
    assert data["prompt"] == "Summarize videos"
    assert data["source_ids"] == ["src1", "src2"]
    assert data["status"] == "completed"
    assert data["content_path"] == content_path
    assert data["error"] is None


@pytest.mark.asyncio
async def test_get_artifact_not_found(client: httpx.AsyncClient):
    resp = await client.get("/artifacts/nonexistent-artifact")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_artifact_content(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    artifact_id = "art-content-001"
    content_path = f"artifacts/{list_id}/{artifact_id}.md"
    artifact_file = tmp_bibilab_home / content_path
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text("# Test Artifact\n\nContent here.", encoding="utf-8")

    from bibilab.db import create_artifact

    await create_artifact(
        artifact_id=artifact_id,
        list_id=list_id,
        name="Test Artifact",
        type="summary",
        prompt="Summarize",
        source_ids=[],
        status="completed",
        content_path=content_path,
    )

    resp = await client.get(f"/artifacts/{artifact_id}/content")
    assert resp.status_code == 200
    assert resp.text == "# Test Artifact\n\nContent here."


@pytest.mark.asyncio
async def test_get_artifact_content_not_found(client: httpx.AsyncClient):
    resp = await client.get("/artifacts/nonexistent-artifact/content")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_patch_artifact_rename(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    artifact_id = "art-rename-001"
    content_path = f"artifacts/{list_id}/{artifact_id}.md"
    artifact_file = tmp_bibilab_home / content_path
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text("# Old Name\n\nContent.", encoding="utf-8")

    from bibilab.db import create_artifact

    await create_artifact(
        artifact_id=artifact_id,
        list_id=list_id,
        name="Old Name",
        type="summary",
        prompt="Summarize",
        source_ids=[],
        status="completed",
        content_path=content_path,
    )

    resp = await client.patch(f"/artifacts/{artifact_id}", json={"name": "New Name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New Name"


@pytest.mark.asyncio
async def test_patch_artifact_not_found(client: httpx.AsyncClient):
    resp = await client.patch("/artifacts/nonexistent-artifact", json={"name": "New Name"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_artifact(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    artifact_id = "art-delete-001"
    content_path = f"artifacts/{list_id}/{artifact_id}.md"
    artifact_file = tmp_bibilab_home / content_path
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text("# To Delete\n\nContent.", encoding="utf-8")

    from bibilab.db import create_artifact

    await create_artifact(
        artifact_id=artifact_id,
        list_id=list_id,
        name="To Delete",
        type="summary",
        prompt="Summarize",
        source_ids=[],
        status="completed",
        content_path=content_path,
    )

    resp = await client.delete(f"/artifacts/{artifact_id}")
    assert resp.status_code == 204
    assert not artifact_file.exists()

    # Verify artifact is gone
    get_resp = await client.get(f"/artifacts/{artifact_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_artifact_not_found(client: httpx.AsyncClient):
    resp = await client.delete("/artifacts/nonexistent-artifact")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_artifact_content_file_missing_ok(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    """DELETE should succeed even if content file is already missing."""
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    artifact_id = "art-no-file-001"
    content_path = f"artifacts/{list_id}/{artifact_id}.md"
    # Don't create the file - it should already be missing

    from bibilab.db import create_artifact

    await create_artifact(
        artifact_id=artifact_id,
        list_id=list_id,
        name="No File",
        type="summary",
        prompt="Summarize",
        source_ids=[],
        status="completed",
        content_path=content_path,
    )

    resp = await client.delete(f"/artifacts/{artifact_id}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_create_artifact_missing_fields(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    # Missing type
    resp = await client.post(
        f"/lists/{list_id}/artifacts",
        json={"prompt": "Summarize", "source_ids": []},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_artifact_stores_ui_lang_in_job_meta(client: httpx.AsyncClient):
    """POST /lists/{list_id}/artifacts stores resolved ui_lang in job meta."""
    import json as json_module

    from bibilab.db import get_job

    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    resp = await client.post(
        f"/lists/{list_id}/artifacts",
        json={
            "type": "summary",
            "prompt": "Summarize the videos",
            "source_ids": ["src1"],
        },
        headers={"X-UI-Lang": "zh"},
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]
    job_row = await get_job(job_id)
    meta = json_module.loads(job_row["meta"])
    assert meta["ui_lang"] == "zh"
