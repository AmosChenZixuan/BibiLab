import json

import httpx
import pytest

from eval import api


def _with_transport(monkeypatch, handler):
    monkeypatch.setattr(api, "transport", httpx.MockTransport(handler))


def test_get_transcript_requests_timestamp_free_view(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["include_time"] = request.url.params.get("include_time")
        return httpx.Response(200, json={"id": "s1", "transcript": "[SPK?] hello"})

    _with_transport(monkeypatch, handler)
    assert api.get_transcript("s1") == "[SPK?] hello"
    assert seen["path"] == "/api/sources/s1"
    assert seen["include_time"] == "false"


def test_call_llm_sends_prompt_and_override(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"text": "graded"})

    _with_transport(monkeypatch, handler)
    out = api.call_llm("judge this", llm={"model": "m"}, timeout=300)
    assert out == "graded"
    assert seen == {"prompt": "judge this", "timeout": 300, "llm": {"model": "m"}}


def test_call_llm_null_override_omits_llm_key(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"text": "ok"})

    _with_transport(monkeypatch, handler)
    api.call_llm("p")
    assert "llm" not in seen


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"detail": {"error": "llm_rate_limit_error"}}, "llm_rate_limit_error"),
        ({"detail": "List not found"}, "List not found"),
    ],
)
def test_error_detail_unwrapped(monkeypatch, body, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=body)

    _with_transport(monkeypatch, handler)
    with pytest.raises(RuntimeError, match=expected):
        api.call_llm("p")


def test_error_non_json_body_falls_back_to_status(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    _with_transport(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="HTTP 502"):
        api.get_lists()


def test_effective_profile_prefers_override(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not hit the backend when an override is set")

    _with_transport(monkeypatch, handler)
    snap = api.effective_profile({"protocol": "openai", "model": "m-x", "api_key": "secret"})
    assert snap.model == "m-x"
    # api_key never lands in snapshots — they're persisted display records.
    assert snap.api_key is None


def test_effective_profile_falls_back_to_backend_config(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/config"
        return httpx.Response(
            200,
            json={"ai": {"protocol": "anthropic", "model": "claude-x", "base_url": None, "api_key": "sk-6e...4bxO"}},
        )

    _with_transport(monkeypatch, handler)
    snap = api.effective_profile(None)
    assert snap.protocol == "anthropic"
    assert snap.model == "claude-x"
    assert snap.api_key is None
