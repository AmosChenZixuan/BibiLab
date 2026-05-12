"""Tests for params_for_expected_hits."""


class TestParamsForExpectedHits:
    def test_one(self):
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import params_for_expected_hits

        p = params_for_expected_hits("one")
        assert p == RetrievalParams(depth_per_source=1, top_k=2)

    def test_few(self):
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import params_for_expected_hits

        p = params_for_expected_hits("few")
        assert p == RetrievalParams(depth_per_source=2, top_k=8)

    def test_many(self):
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import params_for_expected_hits

        p = params_for_expected_hits("many")
        assert p == RetrievalParams(depth_per_source=5, top_k=24)

    def test_unknown_falls_back_to_few(self):
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import params_for_expected_hits

        p = params_for_expected_hits("garbage")
        assert p == RetrievalParams(depth_per_source=2, top_k=8)
