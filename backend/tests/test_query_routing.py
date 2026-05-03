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
async def test_query_classifications_table_exists(tmp_bibilab_home):
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
    from bibilab.models._enums import map_type_to_mode

    assert map_type_to_mode("factual") == "focused"
    assert map_type_to_mode("breadth") == "broad"
    assert map_type_to_mode("analytical") == "focused"
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


def test_parse_response_quoted_word():
    from bibilab.pipeline.route import _parse_response

    assert _parse_response('"breadth"') == "breadth"


def test_parse_response_word_with_explanation():
    from bibilab.pipeline.route import _parse_response

    raw = "breadth\n\nThe query asks to survey across multiple video sources to list out all methods mentioned."
    assert _parse_response(raw) == "breadth"


def test_parse_response_word_with_trailing_period():
    from bibilab.pipeline.route import _parse_response

    assert _parse_response("analytical.") == "analytical"


def test_parse_response_invalid_word():
    from bibilab.pipeline.route import _parse_response

    with pytest.raises(ValueError):
        _parse_response("not a type")


def test_parse_response_multiline_period_explanation():
    from bibilab.pipeline.route import _parse_response

    assert _parse_response("factual.\n\nThis query asks for a specific fact.") == "factual"


def test_parse_response_markdown_bold():
    from bibilab.pipeline.route import _parse_response

    assert _parse_response("**breadth**") == "breadth"


def test_parse_response_markdown_bold_with_explanation():
    from bibilab.pipeline.route import _parse_response

    raw = "**breadth**\n\nThe query asks to find information about a specific entity across multiple sources."
    assert _parse_response(raw) == "breadth"


@pytest.mark.asyncio
async def test_query_classifications_persisted(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, get_db, log_query_classification

    await bootstrap_db()
    await log_query_classification(
        list_id="test-list-id",
        query_text="test query",
        query_type="factual",
        effective_mode="focused",
    )

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT list_id, query_text, query_type, effective_mode FROM query_classifications WHERE list_id = ?",
            ["test-list-id"],
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row["list_id"] == "test-list-id"
    assert row["query_text"] == "test query"
    assert row["query_type"] == "factual"
    assert row["effective_mode"] == "focused"


def test_rerank_min_score_is_none_by_default():
    from bibilab.config import RagConfig

    cfg = RagConfig()
    assert cfg.rerank_min_score is None
