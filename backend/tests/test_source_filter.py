import pytest


class TestExpectedHitsEnum:
    def test_expected_hits_literal_values(self):
        from bibilab.models._enums import ExpectedHits

        assert "one" in ExpectedHits.__args__
        assert "few" in ExpectedHits.__args__
        assert "many" in ExpectedHits.__args__


class TestParamsForExpectedHits:
    def test_one_precision_leaning(self):
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import params_for_expected_hits

        p = params_for_expected_hits("one")
        assert p == RetrievalParams(depth_per_source=1, top_k=2)

    def test_few_default(self):
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import params_for_expected_hits

        p = params_for_expected_hits("few")
        assert p == RetrievalParams(depth_per_source=2, top_k=8)

    def test_many_breadth_leaning(self):
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import params_for_expected_hits

        p = params_for_expected_hits("many")
        assert p == RetrievalParams(depth_per_source=5, top_k=24)

    def test_unknown_falls_back_to_few(self):
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import params_for_expected_hits

        p = params_for_expected_hits("garbage")
        assert p == RetrievalParams(depth_per_source=2, top_k=8)


class TestApplySourceFilter:
    @pytest.mark.asyncio
    async def test_title_contains_case_insensitive_match(self, tmp_bibilab_home):
        from bibilab.db import bootstrap_db, get_db
        from bibilab.pipeline.embed import apply_source_filter

        await bootstrap_db()

        # list1 must exist for FK constraint
        async with get_db() as db:
            await db.execute("INSERT INTO lists (id, name) VALUES (?, ?)", ("list1", "Test List"))
            await db.commit()

        # Insert test sources directly via get_db
        async with get_db() as db:
            await db.execute(
                "INSERT INTO sources (id, video_id, title, list_id, platform) VALUES (?, ?, ?, ?, ?)",
                ("s8", "v8", "第八集 美食推荐", "list1", "test"),
            )
            await db.execute(
                "INSERT INTO sources (id, video_id, title, list_id, platform) VALUES (?, ?, ?, ?, ?)",
                ("s3", "v3", "第3道菜 做法", "list1", "test"),
            )
            await db.commit()

        result = await apply_source_filter(
            source_ids=["s8", "s3"],
            source_filter={"title_contains": "第八集"},
        )
        assert result == ["v8"]

    @pytest.mark.asyncio
    async def test_title_contains_no_match_returns_empty(self, tmp_bibilab_home):
        from bibilab.db import bootstrap_db, get_db
        from bibilab.pipeline.embed import apply_source_filter

        await bootstrap_db()

        async with get_db() as db:
            await db.execute("INSERT INTO lists (id, name) VALUES (?, ?)", ("list1", "Test List"))
            await db.execute(
                "INSERT INTO sources (id, video_id, title, list_id, platform) VALUES (?, ?, ?, ?, ?)",
                ("s8", "v8", "第八集", "list1", "test"),
            )
            await db.commit()

        result = await apply_source_filter(
            source_ids=["s8"],
            source_filter={"title_contains": "不存在的视频"},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_no_filter_returns_none(self, tmp_bibilab_home):
        from bibilab.pipeline.embed import apply_source_filter

        result = await apply_source_filter(source_ids=["s1", "s2"], source_filter=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_filter_returns_empty(self, tmp_bibilab_home):
        from bibilab.pipeline.embed import apply_source_filter

        result = await apply_source_filter(source_ids=["s1"], source_filter={})
        assert result == []
