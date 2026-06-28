"""Tests for bibilab.routers.models."""

from pathlib import Path
from unittest.mock import patch

from bibilab.config import BibilabConfig
from bibilab.routers.models import list_models


async def test_list_models_exposes_only_host_reranker(tmp_bibilab_home: Path):
    """The catalog lists exactly the host-derived reranker variant — never the
    other, which the loader would never use (a download button for it misleads)."""
    cfg = BibilabConfig()

    with patch("bibilab.routers.models.reranker_spec_id", return_value="bge-reranker-base-q"):
        rerankers = [m.id for m in await list_models(cfg) if m.kind == "reranker"]
    assert rerankers == ["bge-reranker-base-q"]

    with patch("bibilab.routers.models.reranker_spec_id", return_value="bge-reranker-base"):
        rerankers = [m.id for m in await list_models(cfg) if m.kind == "reranker"]
    assert rerankers == ["bge-reranker-base"]
