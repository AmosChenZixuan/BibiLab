"""Tests for query routing (Slice 1: classify_query)."""

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_classify_query_returns_factual():
    from bibilab.config import AIConfig, BibilabConfig

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="gpt-4o", api_key="sk-test", base_url="https://api.test"))
    with patch("bibilab.pipeline.route._call_llm", return_value='"factual"') as mock:
        from bibilab.pipeline.route import classify_query

        result = await classify_query("What does video X say about Y?", cfg)
        assert result == "factual"
        mock.assert_called_once()


@pytest.mark.asyncio
async def test_query_classifications_table_exists():
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        rows = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_classifications'")
        result = await rows.fetchall()
        assert len(result) == 1


def test_rag_config_has_query_routing_enabled():
    from bibilab.config import RagConfig

    cfg = RagConfig()
    assert cfg.query_routing_enabled is True


def test_map_type_to_mode():
    from bibilab.pipeline.route import map_type_to_mode

    assert map_type_to_mode("factual") == "focused"
    assert map_type_to_mode("breadth") == "broad"
    assert map_type_to_mode("analytical") == "broad"
    with pytest.raises(ValueError):
        map_type_to_mode("unknown")


@pytest.mark.asyncio
async def test_classify_query_breadth():
    from bibilab.config import AIConfig, BibilabConfig

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="gpt-4o", api_key="sk-test", base_url="https://api.test"))
    with patch("bibilab.pipeline.route._call_llm", return_value='"breadth"'):
        from bibilab.pipeline.route import classify_query

        result = await classify_query("Which videos discuss X?", cfg)
        assert result == "breadth"


@pytest.mark.asyncio
async def test_classify_query_analytical():
    from bibilab.config import AIConfig, BibilabConfig

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="gpt-4o", api_key="sk-test", base_url="https://api.test"))
    with patch("bibilab.pipeline.route._call_llm", return_value='"analytical"'):
        from bibilab.pipeline.route import classify_query

        result = await classify_query("Compare X and Y across videos", cfg)
        assert result == "analytical"


@pytest.mark.asyncio
async def test_classify_query_parse_failure_falls_back_to_factual():
    from bibilab.config import AIConfig, BibilabConfig

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="gpt-4o", api_key="sk-test", base_url="https://api.test"))
    with patch("bibilab.pipeline.route._call_llm", return_value='"not a type"'):
        from bibilab.pipeline.route import classify_query

        result = await classify_query("test query", cfg)
        assert result == "factual"


@pytest.mark.asyncio
async def test_classify_query_llm_failure_falls_back_to_factual():
    from bibilab.config import AIConfig, BibilabConfig

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="gpt-4o", api_key="sk-test", base_url="https://api.test"))
    with patch("bibilab.pipeline.route._call_llm", side_effect=Exception("llm down")):
        from bibilab.pipeline.route import classify_query

        result = await classify_query("test query", cfg)
        assert result == "factual"
