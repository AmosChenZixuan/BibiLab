# Roadmap — Open Questions

### v0

1. ~~Whisper language detection~~ — UI dropdown (`auto / zh / en`), stored in config, applied globally.
2. ~~Note deduplication~~ — `sources` table as dedup source; `?rerun=true` to re-process in-place.
3. ~~List assignment~~ — User always selects a list before ingesting.
4. ~~Digest sync model~~ — Backend writes digest to sources table; web UI is read-only.
5. ~~Chunk strategy~~ — Greedy merge of Whisper segments to ~300 token target.
6. ~~List storage~~ — SQLite `lists` table.
7. ~~Frontend approach~~ — React + TypeScript SPA served by FastAPI.
8. ~~processing_log naming~~ — Renamed to `sources`; table is a mutable catalog, not an immutable log.

### v1

9. ~~**RAG Q&A**~~ — List-scoped multi-turn chat with transcript citations, hybrid search, cross-encoder reranking, a per-query quantile gate, and LLM-driven tool-calling retrieval (the classifier was eliminated). Implemented (epic #193 → redesign #229 → tool-calling epic #285).
10. ~~**Multimodal vision**~~ — Cancelled. No plan to ship; full stack (config + DB column + worker) removed in #405.
11. **Source truth panel** — User-supplied corrections injected into RAG context. Open: stored as annotations on the note file, or a separate overlay table?

### v2

12. **Mindmap generation** — Mermaid output from LLM. Open: generated on-demand or stored alongside digests?
13. **Audio overview** — LLM script + TTS, scoped to list. Open: TTS engine choice (local or cloud)? Does this become a downloadable artifact or a playable inline player?

### v3

14. **YouTube adapter** — Adapter interface is already defined; YouTube-specific resolver and downloader are not implemented. Open: OAuth vs API key vs cookie auth?
15. **Free-text resolver** — Natural language → platform search → user confirmation → bulk ingest. Open: LLM for intent extraction vs heuristic? How to handle ambiguous queries?
