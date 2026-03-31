from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from locus.config import ObsidianConfig
from locus.pipeline.notes import write_overview_note
from locus.vault import get_list_by_id, get_list_name, scan_lists


def _cfg(vault_path: Path) -> ObsidianConfig:
    return ObsidianConfig(vault_path=str(vault_path), locus_folder="Locus")


def _make_list(vault: Path, name: str, list_id: str) -> Path:
    folder = vault / "Locus" / name
    folder.mkdir(parents=True)
    overview = folder / "_overview.md"
    overview.write_text(
        (
            f"---\n"
            f"locus_list_id: {list_id}\n"
            "created_at: 2026-03-31T00:00:00\n"
            "video_count: 0\n"
            "last_updated: 2026-03-31T00:00:00\n"
            f"---\n\n# {name} - Overview\n"
        ),
        encoding="utf-8",
    )
    return folder


def test_scan_lists_empty(tmp_path: Path):
    assert scan_lists(_cfg(tmp_path)) == []


def test_scan_lists_finds_overview(tmp_path: Path):
    _make_list(tmp_path, "ML Course", "list-abc")
    result = scan_lists(_cfg(tmp_path))
    assert len(result) == 1
    assert result[0].id == "list-abc"
    assert result[0].name == "ML Course"


def test_scan_lists_ignores_folder_without_overview(tmp_path: Path):
    (tmp_path / "Locus" / "NoOverview").mkdir(parents=True)
    assert scan_lists(_cfg(tmp_path)) == []


def test_scan_lists_ignores_overview_without_locus_list_id(tmp_path: Path):
    folder = tmp_path / "Locus" / "Bad"
    folder.mkdir(parents=True)
    (folder / "_overview.md").write_text("---\nvideo_count: 0\n---\n", encoding="utf-8")
    assert scan_lists(_cfg(tmp_path)) == []


def test_get_list_by_id_found(tmp_path: Path):
    _make_list(tmp_path, "Physics", "list-xyz")
    result = get_list_by_id("list-xyz", _cfg(tmp_path))
    assert result is not None
    assert result.name == "Physics"


def test_get_list_by_id_not_found(tmp_path: Path):
    assert get_list_by_id("nope", _cfg(tmp_path)) is None


def test_get_list_name(tmp_path: Path):
    _make_list(tmp_path, "Chemistry", "list-chem")
    assert get_list_name("list-chem", _cfg(tmp_path)) == "Chemistry"
    assert get_list_name("missing", _cfg(tmp_path)) is None


def test_write_overview_note_preserves_created_at(tmp_path: Path):
    _make_list(tmp_path, "Course", "list-course")

    write_overview_note("list-course", "Course", [], [], "Outline", _cfg(tmp_path))

    result = scan_lists(_cfg(tmp_path))
    assert len(result) == 1
    assert result[0].created_at == "2026-03-31T00:00:00"


@pytest.fixture()
def tmp_locus_home(tmp_path: Path):
    with patch("locus.config.locus_home", return_value=tmp_path):
        with patch("locus.db.locus_home", return_value=tmp_path):
            with patch("locus.main.locus_home", return_value=tmp_path):
                yield tmp_path


@pytest.fixture()
def client_with_vault(tmp_locus_home: Path, tmp_path: Path):
    from locus.config import load_config, save_config
    from locus.main import create_app

    vault = tmp_path / "vault"
    vault.mkdir()

    cfg = load_config()
    cfg.obsidian.vault_path = str(vault)
    save_config(cfg)

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture()
def created_list_id(client_with_vault: TestClient) -> str:
    response = client_with_vault.post("/lists", json={"name": "ML Course"})
    assert response.status_code == 201
    return response.json()["id"]


def test_create_list_creates_overview(client_with_vault: TestClient, tmp_path: Path):
    response = client_with_vault.post("/lists", json={"name": "ML Course"})
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "ML Course"
    assert "id" in data

    overview = tmp_path / "vault" / "Locus" / "ML Course" / "_overview.md"
    assert overview.exists()
    assert data["id"] in overview.read_text(encoding="utf-8")


def test_create_list_duplicate_name_returns_409(client_with_vault: TestClient):
    client_with_vault.post("/lists", json={"name": "Physics"})
    response = client_with_vault.post("/lists", json={"name": "Physics"})
    assert response.status_code == 409


def test_create_list_existing_folder_without_overview_returns_409(
    client_with_vault: TestClient, tmp_path: Path
):
    (tmp_path / "vault" / "Locus" / "Physics").mkdir(parents=True)

    response = client_with_vault.post("/lists", json={"name": "Physics"})

    assert response.status_code == 409


def test_create_list_empty_name_returns_422(client_with_vault: TestClient):
    response = client_with_vault.post("/lists", json={"name": "  "})
    assert response.status_code == 422


def test_create_list_requires_vault(tmp_locus_home: Path):
    from locus.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        response = client.post("/lists", json={"name": "Test"})

    assert response.status_code == 400


def test_get_lists_returns_vault_lists(client_with_vault: TestClient):
    client_with_vault.post("/lists", json={"name": "Alpha"})
    client_with_vault.post("/lists", json={"name": "Beta"})
    names = [item["name"] for item in client_with_vault.get("/lists").json()]
    assert "Alpha" in names
    assert "Beta" in names


def test_get_lists_empty(client_with_vault: TestClient):
    assert client_with_vault.get("/lists").json() == []


@pytest.mark.asyncio
async def test_get_list_notes_returns_processing_log_rows(
    client_with_vault: TestClient, created_list_id: str
):
    from locus.db import get_db

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO processing_log (
                video_id, platform, list_id, note_path, transcript_path,
                whisper_model, ai_model, vision_enabled, processed_at, settings_snapshot
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BV1abc",
                "bilibili",
                created_list_id,
                "Locus/ML Course/Intro.md",
                "/tmp/transcripts/BV1abc.txt",
                "large-v3",
                "gpt-4o",
                0,
                "2026-03-30T12:00:00+00:00",
                "{}",
            ),
        )
        await db.commit()

    response = client_with_vault.get(f"/lists/{created_list_id}/notes")
    assert response.status_code == 200
    assert response.json() == [
        {
            "video_id": "BV1abc",
            "note_path": "Locus/ML Course/Intro.md",
            "processed_at": "2026-03-30T12:00:00+00:00",
            "platform": "bilibili",
        }
    ]


@pytest.mark.asyncio
async def test_delete_list_rejects_active_jobs(client_with_vault: TestClient, created_list_id: str):
    from locus.db import create_job

    await create_job(
        type="video",
        source_url="https://www.bilibili.com/video/BV1active",
        platform="bilibili",
        meta={"video_id": "BV1active", "list_id": created_list_id},
    )

    response = client_with_vault.delete(f"/lists/{created_list_id}")
    assert response.status_code == 409
    assert response.json()["detail"] == "Cannot delete a list with active jobs"


def test_delete_list_removes_vault_folder(client_with_vault: TestClient, tmp_path: Path):
    list_id = client_with_vault.post("/lists", json={"name": "ToDelete"}).json()["id"]
    response = client_with_vault.delete(f"/lists/{list_id}")
    assert response.status_code == 204
    assert not (tmp_path / "vault" / "Locus" / "ToDelete").exists()


@pytest.mark.asyncio
async def test_delete_list_cascades_processing_log(
    client_with_vault: TestClient, tmp_locus_home: Path
):
    import aiosqlite

    response = client_with_vault.post("/lists", json={"name": "ToDelete"})
    assert response.status_code == 201
    list_id = response.json()["id"]

    db_path = tmp_locus_home / "locus.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO processing_log
              (video_id, platform, list_id, note_path, transcript_path,
               whisper_model, ai_model, vision_enabled, processed_at, settings_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BV1test",
                "bilibili",
                list_id,
                None,
                None,
                "large-v3",
                "gpt-4o",
                0,
                "2026-01-01T00:00:00+00:00",
                "{}",
            ),
        )
        await db.commit()

    response = client_with_vault.delete(f"/lists/{list_id}")
    assert response.status_code == 204

    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT 1 FROM processing_log WHERE list_id=?", (list_id,)) as cur:
            row = await cur.fetchone()
    assert row is None


def test_delete_list_not_found(client_with_vault: TestClient):
    response = client_with_vault.delete("/lists/nonexistent")
    assert response.status_code == 404
