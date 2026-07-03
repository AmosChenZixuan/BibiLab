"""HTTP client for the backend — the only module that talks to it.

Backend errors arrive as ``{"detail": {"error": "<code>"}}`` (classified
pipeline/LLM failures) or ``{"detail": "<message>"}``; both unwrap into a
RuntimeError whose message is the code/message, so callers' existing
except-Exception → error-string paths keep producing readable errors.
"""

from __future__ import annotations

import httpx

from eval.config import get_backend_url
from eval.models import ProfileSnapshot

# Test seams: fixtures assign httpx.MockTransport here; None = real network.
transport: httpx.BaseTransport | None = None
async_transport: httpx.AsyncBaseTransport | None = None

# Chat turns and bare LLM calls run for minutes on weak local models; the
# server caps llm timeout at 600s, so the client waits a little past that.
_TIMEOUT = httpx.Timeout(660.0, connect=10.0)


def _detail(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, dict) and "error" in detail:
        return str(detail["error"])
    if detail:
        return str(detail)
    return f"HTTP {resp.status_code}"


def _checked(resp: httpx.Response) -> httpx.Response:
    if resp.status_code >= 400:
        raise RuntimeError(_detail(resp))
    return resp


def _client() -> httpx.Client:
    return httpx.Client(base_url=get_backend_url(), timeout=_TIMEOUT, transport=transport)


def get_lists() -> list[dict]:
    with _client() as c:
        return _checked(c.get("/api/lists")).json()


def get_sources(list_id: str) -> list[dict]:
    with _client() as c:
        return _checked(c.get(f"/api/lists/{list_id}/sources")).json()


def get_transcript(source_id: str) -> str:
    """Timestamp-free speaker turns — the view case-generation prompts expect."""
    with _client() as c:
        resp = _checked(c.get(f"/api/sources/{source_id}", params={"include_time": "false"}))
    return resp.json()["transcript"]


def get_backend_ai() -> dict:
    """The backend's own AI config (api_key masked by the endpoint — treat it
    as display-only, never send it back as an override)."""
    with _client() as c:
        return _checked(c.get("/api/config")).json()["ai"]


def effective_profile(llm: dict | None) -> ProfileSnapshot:
    """Record of which LLM actually serves the calls — the override if set,
    else the backend's own config. api_key never lands in snapshots (they're
    display records persisted into run/grade JSON)."""
    source = llm if llm else get_backend_ai()
    fields = {k: v for k, v in source.items() if k in ("protocol", "model", "base_url") and v is not None}
    return ProfileSnapshot.model_validate(fields)


def call_llm(prompt: str, llm: dict | None = None, timeout: int = 120) -> str:
    """Bare LLM call through the backend's own client — byte-identical provider
    requests without any LLM SDK on this side."""
    body: dict = {"prompt": prompt, "timeout": timeout}
    if llm:
        body["llm"] = llm
    with _client() as c:
        return _checked(c.post("/api/eval/llm", json=body)).json()["text"]


async def run_chat(query: str, list_id: str, llm: dict | None = None, language: str | None = None) -> dict:
    """One stateless RAG chat turn; sync JSON, no SSE."""
    body: dict = {"query": query, "list_id": list_id}
    if llm:
        body["llm"] = llm
    if language:
        body["language"] = language
    async with httpx.AsyncClient(base_url=get_backend_url(), timeout=_TIMEOUT, transport=async_transport) as c:
        return _checked(await c.post("/api/eval/run_chat", json=body)).json()
