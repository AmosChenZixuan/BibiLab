"""Tests for citation registry accumulation and dedup."""

from bibilab.pipeline.chat_tools import CitationRegistryEntry


class TestEntry:
    def test_create(self):
        e = CitationRegistryEntry(index=1, source_id="s1")
        assert e.index == 1
        assert e.chunk_ids == set()

    def test_chunk_ids_dedup(self):
        e = CitationRegistryEntry(index=1, source_id="s1")
        e.chunk_ids.add("c1")
        e.chunk_ids.add("c1")
        assert e.chunk_ids == {"c1"}


class TestAccumulation:
    def setup_method(self):
        self._chunk_counter = 0

    def _simulate(self, registry, source_map, video_ids_sources, video_ids_chunks):
        next_idx = max((e.index for e in registry.values()), default=0) + 1
        for vid in video_ids_sources:
            sid = source_map.get(vid)
            if sid and sid not in registry:
                registry[sid] = CitationRegistryEntry(index=next_idx, source_id=sid)
                next_idx += 1
        for vid in video_ids_chunks:
            sid = source_map.get(vid)
            if sid and sid in registry:
                registry[sid].chunk_ids.add(f"c{self._chunk_counter}")
                self._chunk_counter += 1
        return registry

    def test_first_retrieve_1_to_n(self):
        r = {}
        sm = {"va": "sa", "vb": "sb"}
        self._simulate(r, sm, ["va", "vb"], ["va", "vb"])
        assert r["sa"].index == 1
        assert r["sb"].index == 2

    def test_second_retrieve_dedup_extends(self):
        r = {}
        sm = {"va": "sa", "vb": "sb", "vc": "sc"}
        self._simulate(r, sm, ["va", "vb"], ["va"])
        assert r["sa"].index == 1
        assert r["sb"].index == 2
        self._simulate(r, sm, ["vb", "vc"], ["vb", "vc"])
        assert r["sb"].index == 2
        assert r["sc"].index == 3

    def test_chunk_ids_accumulate_across_calls(self):
        r = {}
        sm = {"va": "sa"}
        self._simulate(r, sm, ["va"], ["va"])
        self._simulate(r, sm, ["va"], ["va"])
        assert "c0" in r["sa"].chunk_ids
        assert "c1" in r["sa"].chunk_ids

    def test_empty_retrieve_preserves_registry(self):
        r = {}
        sm = {"va": "sa"}
        self._simulate(r, sm, ["va"], ["va"])
        self._simulate(r, sm, [], [])
        assert len(r) == 1
        assert r["sa"].index == 1
