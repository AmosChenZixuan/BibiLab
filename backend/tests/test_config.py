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
