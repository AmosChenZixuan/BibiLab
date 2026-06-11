from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bibilab.pipeline._shared import _call_llm


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text


def safe_call_llm(prompt: str, ai_cfg: Any, *, llm_timeout: int = 120
                  ) -> tuple[str | None, str | None]:
    """Run one sync LLM call, turning any failure into (None, err_str).

    A timeout / API error must not crash the surrounding step — the eval
    pipeline is partial-success by design (per #525 hard-stop lesson).
    Returns (raw | None, error | None). `_call_llm` is module-level so
    tests can monkeypatch `eval._utils._call_llm` to swap both extraction
    and phrasing paths at once.
    """
    try:
        return (_call_llm(prompt, ai_cfg, llm_timeout=llm_timeout), None)
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")
