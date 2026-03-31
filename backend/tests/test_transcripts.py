from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def tmp_locus_home(tmp_path: Path):
    with patch("locus.config.locus_home", return_value=tmp_path):
        with patch("locus.db.locus_home", return_value=tmp_path):
            with patch("locus.main.locus_home", return_value=tmp_path):
                with patch("locus.routers.transcripts.locus_home", return_value=tmp_path):
                    yield tmp_path


@pytest.fixture()
def client(tmp_locus_home: Path):
    from locus.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_get_transcript_returns_paginated_lines(client: TestClient, tmp_locus_home: Path):
    transcript_dir = tmp_locus_home / "transcripts"
    transcript_dir.mkdir(exist_ok=True)
    transcript_path = transcript_dir / "BV1transcript.txt"
    transcript_path.write_text(
        "\n".join(
            [
                "[00:00:00] line 1",
                "[00:00:05] line 2",
                "[00:00:10] line 3",
                "[00:00:15] line 4",
                "[00:00:20] line 5",
                "[00:00:25] line 6",
            ]
        ),
        encoding="utf-8",
    )

    response = client.get("/transcripts/BV1transcript?offset=1&limit=3")
    assert response.status_code == 200
    assert response.json() == {
        "video_id": "BV1transcript",
        "total_lines": 6,
        "offset": 1,
        "lines": [
            "[00:00:05] line 2",
            "[00:00:10] line 3",
            "[00:00:15] line 4",
        ],
    }


def test_get_transcript_returns_404_when_missing(client: TestClient):
    response = client.get("/transcripts/does-not-exist")
    assert response.status_code == 404
