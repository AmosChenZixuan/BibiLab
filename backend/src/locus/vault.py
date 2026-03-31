"""Vault-backed list discovery and overview writing."""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from locus.config import ObsidianConfig

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
_FIELD_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)


def parse_frontmatter(text: str) -> dict[str, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    return dict(_FIELD_RE.findall(match.group(1)))


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    return dict(_FIELD_RE.findall(match.group(1))), text[match.end() :]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class ListMeta:
    id: str
    name: str
    path: Path
    created_at: str


def locus_dir(cfg: ObsidianConfig) -> Path:
    return Path(cfg.vault_path) / cfg.locus_folder


def write_list_overview(
    overview_path: Path,
    *,
    list_id: str,
    list_name: str,
    created_at: str,
) -> None:
    """Write or update a list _overview.md, preserving existing body and created_at."""
    existing_fm: dict[str, str] = {}
    body = f"# {list_name} - Overview\n"
    if overview_path.exists():
        existing_fm, body = _split_frontmatter(overview_path.read_text(encoding="utf-8"))
        if not body.strip():
            body = f"# {list_name} - Overview\n"

    created = existing_fm.get("created_at") or created_at or _now()
    content = (
        "---\n"
        f"locus_list_id: {list_id}\n"
        f"created_at: {created}\n"
        f"video_count: {existing_fm.get('video_count', '0')}\n"
        f"last_updated: {existing_fm.get('last_updated', created)}\n"
        f"---\n\n{body}"
    )
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = overview_path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, overview_path)


def scan_lists(cfg: ObsidianConfig) -> list[ListMeta]:
    """Return all lists discovered in the vault."""
    ld = locus_dir(cfg)
    if not ld.exists():
        return []

    results: list[ListMeta] = []
    for overview in sorted(ld.glob("*/_overview.md")):
        frontmatter = parse_frontmatter(overview.read_text(encoding="utf-8"))
        list_id = frontmatter.get("locus_list_id", "").strip()
        if not list_id:
            continue
        results.append(
            ListMeta(
                id=list_id,
                name=overview.parent.name,
                path=overview.parent,
                created_at=frontmatter.get("created_at", "").strip(),
            )
        )
    return results


def get_list_by_id(list_id: str, cfg: ObsidianConfig) -> ListMeta | None:
    return next((item for item in scan_lists(cfg) if item.id == list_id), None)


def get_list_name(list_id: str, cfg: ObsidianConfig) -> str | None:
    item = get_list_by_id(list_id, cfg)
    return item.name if item else None
