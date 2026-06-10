"""db package — flat per-table SQL modules + connection bootstrap.

SQL helpers live in submodules (``bibilab.db.sources``, ``bibilab.db.lists``,
etc.) and must be imported from there directly. This package root exposes
only the 4 moved-out domain functions re-exported lazily below, to break
the ``db → worker/chat_runs → db`` partial-init cycle.
"""


def __getattr__(name):
    """Lazy re-exports of moved-out domain functions (PEP 562).

    Eager imports here would form a partial-init cycle (db → worker/chat_runs
    → db) that breaks when any module imports ``bibilab.worker`` before
    ``bibilab.db``. Lazy resolution defers the import until first attribute
    access, after the db package is fully initialized — same pattern
    ``chat_runs.py`` uses locally for the same kind of cycle.
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
