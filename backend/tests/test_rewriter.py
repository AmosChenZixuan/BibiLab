import pytest
from pydantic import ValidationError

from bibilab.pipeline.rewriter import RewriterIntent


class TestRewriterIntentInvariants:
    def test_retrieve_false_with_all_null_fields_is_valid(self):
        intent = RewriterIntent(retrieve=False)
        assert intent.retrieve is False
        assert intent.query is None
        assert intent.mode is None

    def test_retrieve_false_with_query_raises(self):
        with pytest.raises(ValidationError, match="retrieve=false requires"):
            RewriterIntent(retrieve=False, query="hello")

    def test_retrieve_false_with_mode_raises(self):
        with pytest.raises(ValidationError, match="retrieve=false requires"):
            RewriterIntent(retrieve=False, mode="narrow")

    def test_retrieve_false_with_sequence_number_raises(self):
        with pytest.raises(ValidationError, match="retrieve=false requires"):
            RewriterIntent(retrieve=False, sequence_number=5)

    def test_retrieve_true_with_query_and_mode_is_valid(self):
        intent = RewriterIntent(retrieve=True, query="multiplexing", mode="narrow")
        assert intent.query == "multiplexing"
        assert intent.mode == "narrow"

    def test_retrieve_true_without_query_raises(self):
        with pytest.raises(ValidationError, match="retrieve=true requires"):
            RewriterIntent(retrieve=True, mode="narrow")

    def test_retrieve_true_without_mode_raises(self):
        with pytest.raises(ValidationError, match="retrieve=true requires"):
            RewriterIntent(retrieve=True, query="x")

    def test_invalid_mode_value_raises(self):
        with pytest.raises(ValidationError):
            RewriterIntent(retrieve=True, query="x", mode="few")  # legacy value
