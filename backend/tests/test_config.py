"""Tests for BibilabConfig schema shape after #405 cleanup.

These tests assert the absence of fields that #405 removes from the schema.
They are written before the schema change so that they fail on the current
schema, then pass once the schema change lands.
"""


def test_vision_config_removed():
    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    assert not hasattr(cfg, "vision")
    assert "vision" not in cfg.model_dump()


def test_accounts_bilibili_no_last_verified():
    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    bilibili = cfg.accounts.bilibili
    assert not hasattr(bilibili, "last_verified")
    assert "last_verified" not in bilibili.model_dump()


def test_ai_no_transcript_char_limit():
    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    assert not hasattr(cfg.ai, "transcript_char_limit")
    assert "transcript_char_limit" not in cfg.ai.model_dump()


def test_rag_no_chunk_pause_threshold():
    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    assert not hasattr(cfg.rag, "chunk_pause_threshold")
    assert "chunk_pause_threshold" not in cfg.rag.model_dump()


def test_no_transcript_collection_name():
    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    assert not hasattr(cfg, "transcript_collection_name")
    assert "transcript_collection_name" not in cfg.model_dump()
