"""Tests for opt-in per-message LLM prompt-trace dump (#393)."""


def test_rag_config_default_debug_prompts_is_false():
    """Off by default — opt-in flag, zero behavior change for existing users."""
    from bibilab.config import RagConfig

    cfg = RagConfig()
    assert cfg.debug_prompts is False
