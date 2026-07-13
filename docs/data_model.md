# Data Model

Column-level reference for every SQLite table in `~/.bibilab/bibilab.db`. Authoritative DDL lives in `backend/src/bibilab/db.py` (`bootstrap_db`); the invariants an AI must not break when touching these tables are summarized in `backend/CLAUDE.md`. Storage layout and path conventions are in the root `CLAUDE.md`.

### `lists`

| Column | Notes |
|---|---|
| `thumbnail_source_id` | FK to `sources.id`, nullable |

### `jobs` — ephemeral queue

| Column | Notes |
|---|---|
| `type` | `"ingest"` \| `"model_download"` \| `"artifact"` |
| `status` | `queued` → `downloading` → `transcribing` → `processing` → `done` \| `failed` \| `needs_auth` |
| `progress` | 0–100 |
| `error` | Error message, nullable |
| `meta` | JSON blob: `{ video_id, list_id, title, cover_url, duration_seconds, uploader, source_url, platform, ui_lang }` (ingest); `{ source_id, list_id, source_title, ui_lang }` (digest rerun) |

### `sources` — active video catalog

| Column | Notes |
|---|---|
| `video_id` | Platform-native ID (e.g. `bvid`) |
| `title`, `summary`, `keywords` | `title` is platform metadata; `summary`/`keywords` columns DROPPED — sections are the sole digest store |
| `language`, `uploader`, `duration_seconds` | Video metadata |
| `whisper_model`, `ai_model` | Processing config at ingest time |
| `cover_url`, `processed_at`, `settings_snapshot` | Ingest-time state |
| `series_name`, `sequence_number`, `season_number` | Facet metadata extracted by LLM digest (nullable) |

### `artifacts` — Lab-generated content

| Column | Notes |
|---|---|
| `name` | User-visible name (initial = type label, renameable) |
| `type` | `"brief"` \| `"study_guide"` \| `"blog_post"` \| `"custom_report"` \| `"mind_map"` (JSON tree, viewer-rendered) \| `"chat_message"` (saved chat turn) |
| `prompt` | Exact prompt string submitted |
| `source_ids` | JSON array of source UUIDs |
| `status` | `generating` → `done` \| `failed` |
| `content_path` | Relative path, e.g. `artifacts/{id}.md`, nullable |
| `error` | Error message, nullable |

### `conversations` — one per list

| Column | Notes |
|---|---|
| `list_id` | FK to `lists.id`, UNIQUE, ON DELETE CASCADE |
| `summary` | Rolling LLM-generated summary, nullable |
| `active_stream_message_id` | FK to `messages.id`, nullable; points to the currently streaming assistant message |
| `updated_at` | ISO timestamp |

### `messages` — chat history

| Column | Notes |
|---|---|
| `conversation_id` | FK to `conversations.id`, ON DELETE CASCADE, indexed |
| `role` | `"user"` \| `"assistant"` \| `"tool"` |
| `content` | Message text |
| `status` | In-flight: user `"pending"`, assistant `"streaming"`. Terminal: `"done"` \| `"failed"` \| `"cancelled"`. Both rows of a turn flip to the **same** terminal status atomically (`update_turn_terminal`); a turn is visible to LLM replay + compaction iff both are `"done"`. UI history is unfiltered (renders 已停止/重试); the LLM snapshot filters to `"done"` inline (not in `get_recent_messages`). |
| `error` | Error code (e.g. `llm_rate_limit_error`, `internal_error`), nullable; set on producer failure or server restart sweep; mapped by `_classify_llm_error()` from SDK exceptions; frontend resolves via `chat.errors.*` i18n keys |
| `metadata` | JSON blob, nullable: `{"content_blocks": [...], "rag": {"calls": [...]}}` — full field shape in `docs/chat_architecture.md`. Set by `run_chat_turn` post-stream. No migration for legacy rows — the ledger renders best-effort from whatever fields exist. |
| `tool_blocks` | JSON array of the assistant turn's tool-use/tool-result blocks, nullable; written with the terminal flip (`update_turn_terminal`). Consumed by `expand_message_for_provider` (provider replay) and `reseed_citation_registry` (cross-turn `[N]` reseed). Tool-call data lives here, **not** in `metadata`. |

### `transcript_segments` — punctuated sentence segments

| Column | Notes |
|---|---|
| `source_id` | FK to `sources.id`, ON DELETE CASCADE |
| `seq` | Segment order within a source |
| `start_s` | Start time in seconds |
| `end_s` | End time in seconds |
| `speaker` | Speaker label (e.g. `SPK_0`), nullable |
| `text` | Punctuated sentence text |

Index: `idx_segments_source` on `(source_id, seq)`. Replaces the on-disk `transcripts/{source_id}.txt` (P2).

### `sections` — bounded sub-source spans

| Column | Notes |
|---|---|
| `source_id` | FK to `sources.id`, ON DELETE CASCADE |
| `seq` | Section order within a source |
| `seg_start` | First segment index in the source |
| `seg_end` | Last segment index in the source (inclusive) |
| `token_count` | Sum of `count_tokens(seg.text)` across the section's segments |
| `timestamp_start` | Wall-clock seconds at `segments[seg_start].start` |
| `timestamp_end` | Wall-clock seconds at `segments[seg_end].end` |
| `summary` | LLM refine-summary; always populated (written with the section row) |
| `keywords` | LLM keywords (JSON array); always populated |

Index: `idx_sections_source` on `(source_id, seq)`. Chunks produced per-section nest physically (a chunk's `[seg_start, seg_end]` is fully contained in exactly one section's). Production write path is `_exec_write_sections` (called from `write_source_with_segments` in the same transaction as source + segments); `section_digests` is required whenever `sections` is provided, so every section row carries a summary.

### `chunks_fts` — FTS5 virtual table

BM25-ranked full-text index over transcript chunks. Populated by `populate_fts()` during the embed stage; cleared per-source on re-ingest. Columns: `content`, `pinyin` (toneless-pinyin syllable-bigram index for CJK, built by `_pinyin_index_tokens`), `source_id`, `video_title`, `timestamp_start`, `timestamp_end`, `chunk_id`, `seg_start`, `seg_end`. Query via `query_fts_rows()` in `db.py`.
