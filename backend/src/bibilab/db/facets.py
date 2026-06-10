"""Source-facet writers (digest path + manual-edit path).

Both writers target the ``sources`` table's facet columns
(``series_name``, ``sequence_number``, ``season_number``) and have
deliberately different semantics:

- ``apply_digest_facets`` (COALESCE-preserve) — the digest LLM best-effort
  guess; a None for an unspecified facet must not clobber a previously
  stored value.
- ``update_source_facets`` (replace; explicit None clears) — the manual
  user edit; an explicit None is the user's choice and replaces the
  stored value.
"""

from __future__ import annotations

from bibilab.db.connection import _now, get_db


async def apply_digest_facets(
    source_id: str,
    series_name: str | None = None,
    sequence_number: int | None = None,
    season_number: int | None = None,
    bump_processed_at: bool = True,
) -> None:
    # Mirrors the facets the digest step extracted onto the source row (summaries
    # live per-section in the sections table; the source carries facets only).
    # Reruns pass bump_processed_at=False: processed_at anchors list ordering
    # (ORDER BY processed_at ASC in list_sources), and a digest rerun shouldn't
    # move the source within the list.
    sets = [
        "series_name=COALESCE(?, series_name)",
        "sequence_number=COALESCE(?, sequence_number)",
        "season_number=COALESCE(?, season_number)",
    ]
    params: list[object] = [series_name, sequence_number, season_number]
    if bump_processed_at:
        sets.append("processed_at=?")
        params.append(_now())
    params.append(source_id)
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE sources SET {', '.join(sets)} WHERE id=?",
            params,
        )
        if cursor.rowcount == 0:
            raise LookupError(source_id)
        await db.commit()


# Manual-edit facet writer. Replace semantics (explicit None clears), distinct
# from apply_digest_facets's COALESCE-preserve.
_FACET_WRITE_COLUMNS = ("series_name", "sequence_number", "season_number")


async def update_source_facets(source_id: str, **fields: object) -> None:
    cols = [c for c in _FACET_WRITE_COLUMNS if c in fields]
    if not cols:
        return
    # Column names come from the fixed allowlist above (never user input);
    # values stay parameterized — db's no-f-string-values rule holds.
    set_clause = ", ".join(f"{c}=?" for c in cols)
    params = [fields[c] for c in cols]
    params.append(source_id)
    async with get_db() as db:
        cursor = await db.execute(f"UPDATE sources SET {set_clause} WHERE id=?", params)
        if cursor.rowcount == 0:
            # Source vanished between the router's existence check and this
            # write (TOCTOU). Don't commit a no-op as success.
            raise LookupError(source_id)
        await db.commit()
