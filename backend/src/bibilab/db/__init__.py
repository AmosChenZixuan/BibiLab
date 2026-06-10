"""db package — flat per-table SQL modules + connection bootstrap.

The original ``db.py`` was decomposed into this package. The 3 domain
functions it used to host (``reset_stuck_jobs``, ``create_user_and_assistant_atomic``,
``update_turn_terminal``) have moved to their proper homes
(``bibilab.worker`` and ``bibilab.pipeline.chat_runs``); they are
re-exported here so the pre-split ``from bibilab.db import …`` surface
keeps working. New code should import from the canonical home directly.
"""

from bibilab.db.artifacts import (
    create_artifact,
    delete_artifact,
    delete_artifacts_for_list,
    get_artifact,
    get_artifacts_for_list,
    update_artifact_name,
)
from bibilab.db.connection import _now, bootstrap_db, get_db, get_db_path, source_exists_sync
from bibilab.db.conversations import (
    delete_conversation,
    get_conversation,
    get_conversation_by_list,
    get_or_create_conversation,
    set_active_stream,
)
from bibilab.db.facets import apply_digest_facets, update_source_facets
from bibilab.db.fts import (
    _cjk_bigram_tokens,
    _cjk_query_tokens,
    _cjk_runs,
    _collapse_cjk_whitespace,
    _escape_fts_query,
    _fts_quote_token,
    _pinyin_index_tokens,
    _pinyin_tokens,
    _tokenize_cjk,
    clear_fts_for_list,
    query_fts_rows,
)
from bibilab.db.jobs import (
    create_job,
    delete_job,
    get_job,
    get_jobs_for_video_ids,
    get_pending_jobs,
    get_source_video_ids,
    list_jobs,
    parse_job_meta,
    update_job_meta,
    update_job_status,
)
from bibilab.db.lists import (
    create_list,
    delete_list,
    get_all_lists,
    get_list,
    get_list_with_display,
    update_list_name,
    update_list_thumbnail,
)
from bibilab.db.messages import (
    IN_FLIGHT_ASST_STATUS,
    IN_FLIGHT_MESSAGE_STATUSES,
    IN_FLIGHT_USER_STATUS,
    VISIBLE_MESSAGE_STATUS,
    assert_message_in_list,
    compress_conversation,
    get_message_count,
    get_messages_beyond_window,
    get_recent_messages,
)
from bibilab.db.sections import (
    get_section_ranges,
    get_sections,
    rows_to_sections,
    update_section_summaries,
)
from bibilab.db.segments import (
    get_segments_for_ranges,
    get_transcript_segments,
    rows_to_segments,
)
from bibilab.db.sources import (
    _exec_write_sections,
    _exec_write_source,
    _exec_write_transcript_segments,
    _in_placeholders,
    delete_source,
    delete_sources_for_list,
    get_source,
    get_source_facets,
    get_sources_for_list,
    write_source_with_segments,
    write_transcript_segments,
)
from bibilab.models.jobs import JobStatus

# Moved-out domain functions are re-exported lazily via __getattr__ below.
# Eager imports here would form a partial-init cycle (db → worker/chat_runs → db)
# that breaks when any module imports bibilab.worker before bibilab.db. Lazy
# resolution defers the import until first attribute access, after the db
# package is fully initialized — same pattern chat_runs.py uses locally.


__all__ = [
    # connection
    "_now",
    "bootstrap_db",
    "get_db",
    "get_db_path",
    "source_exists_sync",
    # lists
    "create_list",
    "delete_list",
    "get_all_lists",
    "get_list",
    "get_list_with_display",
    "update_list_name",
    "update_list_thumbnail",
    # sources
    "_exec_write_sections",
    "_exec_write_source",
    "_exec_write_transcript_segments",
    "_in_placeholders",
    "apply_digest_facets",
    "delete_source",
    "delete_sources_for_list",
    "get_segments_for_ranges",
    "get_source",
    "get_source_facets",
    "get_sources_for_list",
    "rows_to_segments",
    "update_source_facets",
    "write_source_with_segments",
    "write_transcript_segments",
    # sections
    "get_section_ranges",
    "get_sections",
    "rows_to_sections",
    "update_section_summaries",
    # segments
    "get_transcript_segments",
    # conversations
    "delete_conversation",
    "get_conversation",
    "get_conversation_by_list",
    "get_or_create_conversation",
    "set_active_stream",
    # messages
    "IN_FLIGHT_ASST_STATUS",
    "IN_FLIGHT_MESSAGE_STATUSES",
    "IN_FLIGHT_USER_STATUS",
    "VISIBLE_MESSAGE_STATUS",
    "assert_message_in_list",
    "compress_conversation",
    "get_message_count",
    "get_messages_beyond_window",
    "get_recent_messages",
    # artifacts
    "create_artifact",
    "delete_artifact",
    "delete_artifacts_for_list",
    "get_artifact",
    "get_artifacts_for_list",
    "update_artifact_name",
    # jobs
    "JobStatus",
    "create_job",
    "delete_job",
    "get_job",
    "get_jobs_for_video_ids",
    "get_pending_jobs",
    "get_source_video_ids",
    "list_jobs",
    "parse_job_meta",
    "update_job_meta",
    "update_job_status",
    # fts
    "_cjk_bigram_tokens",
    "_cjk_query_tokens",
    "_cjk_runs",
    "_collapse_cjk_whitespace",
    "_escape_fts_query",
    "_fts_quote_token",
    "_pinyin_index_tokens",
    "_pinyin_tokens",
    "_tokenize_cjk",
    "clear_fts_for_list",
    "query_fts_rows",
    # Moved-out domain functions (re-exported for backward compat via __getattr__).
    "ActiveStreamConflict",
    "create_user_and_assistant_atomic",
    "reset_stuck_jobs",
    "update_turn_terminal",
]


def __getattr__(name):
    """Lazy re-exports of moved-out domain functions (PEP 562).

    The moved functions live in ``bibilab.worker`` and
    ``bibilab.pipeline.chat_runs``; eager imports here would form a partial-init
    cycle (db → worker/chat_runs → db) that breaks when any module imports
    ``bibilab.worker`` before ``bibilab.db``. Lazy resolution defers the import
    until first attribute access, after the db package is fully initialized —
    same pattern ``chat_runs.py`` uses locally for the same kind of cycle.
    """
    if name in ("ActiveStreamConflict", "create_user_and_assistant_atomic", "update_turn_terminal"):
        from bibilab.pipeline.chat_runs import (  # noqa: F401
            ActiveStreamConflict,
            create_user_and_assistant_atomic,
            update_turn_terminal,
        )

        return locals()[name]
    if name == "reset_stuck_jobs":
        from bibilab.worker import reset_stuck_jobs

        return reset_stuck_jobs
    raise AttributeError(f"module 'bibilab.db' has no attribute {name!r}")
