"""Tests for cross-turn retrieval reuse classifier (#280)."""

import pytest

from bibilab.pipeline.chat_tools import ReuseAction, decide_reuse


class TestDecideReuse:
    """Unit tests for decide_reuse() decision matrix."""

    # --- Trivial message guard ---

    @pytest.mark.parametrize("msg", ["1", "12", "99"])
    def test_pure_digits_are_trivial(self, msg):
        decision = decide_reuse(msg, "prior question about physics")
        assert decision.action == ReuseAction.TRIVIAL

    @pytest.mark.parametrize(
        "msg",
        [
            "",
            " ",
            "  ",
            "a",
            "ab",
            " a ",
            " . ",
            "是吗",  # 2 Chinese chars < 3 non-ws
        ],
    )
    def test_short_messages_are_trivial(self, msg):
        decision = decide_reuse(msg, "prior question about physics")
        assert decision.action == ReuseAction.TRIVIAL

    @pytest.mark.parametrize(
        "msg",
        [
            "否",
            "ok",
            "OK",
            "Ok",
            "continue",
            "好的",
            "好",
            "是",
            "对",
            "yes",
            "no",
            "thanks",
            "谢谢",
            "k",
            "kk",
        ],
    )
    def test_stopwords_are_trivial(self, msg):
        decision = decide_reuse(msg, "prior question about physics")
        assert decision.action == ReuseAction.TRIVIAL

    def test_trivial_path_includes_note(self):
        decision = decide_reuse("1", "prior question")
        assert decision.note is not None
        assert "acknowledgment" in decision.note.lower()

    @pytest.mark.parametrize("msg", ["why", "what", "tell me more"])
    def test_non_trivial_short_messages_pass_guard(self, msg):
        decision = decide_reuse(msg, "prior question about physics")
        assert decision.action != ReuseAction.TRIVIAL

    # --- No prior context ---

    def test_no_prior_message_forces_fresh(self):
        decision = decide_reuse("tell me about quantum physics", None)
        assert decision.action == ReuseAction.FORCE_FRESH

    # --- Cosine similarity (requires embedding model) ---

    def test_identical_messages_keep(self):
        msg = "tell me about quantum mechanics in episode 3"
        decision = decide_reuse(msg, msg)
        assert decision.action == ReuseAction.KEEP

    def test_unrelated_messages_force_fresh(self):
        decision = decide_reuse(
            "what is the capital of France",
            "explain quantum mechanics in detail with equations",
        )
        assert decision.action == ReuseAction.FORCE_FRESH

    def test_paraphrase_same_topic_keep(self):
        decision = decide_reuse(
            "what did episode 3 say about quantum mechanics",
            "can you tell me about quantum mechanics in episode 3",
        )
        assert decision.action == ReuseAction.KEEP

    def test_deepening_followup_keep(self):
        decision = decide_reuse(
            "what are the key principles of quantum mechanics",
            "tell me about quantum mechanics in episode 3",
        )
        assert decision.action == ReuseAction.KEEP

    def test_topic_switch_force_fresh(self):
        decision = decide_reuse(
            "tell me about python programming",
            "explain the plot of episode 5 in detail",
        )
        assert decision.action == ReuseAction.FORCE_FRESH
