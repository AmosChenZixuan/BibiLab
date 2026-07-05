"""Integration tests for POST /eval/llm.

Bare LLM call routed through the backend's own `_call_llm`, so the eval
framework's generation/grading requests are byte-identical to the backend's
provider calls without any LLM SDK on the eval side.
"""

import pytest

from bibilab.config import get_config

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_eval_llm_happy_path_uses_backend_config(client, mock_call_llm):
    mock_call_llm.return_value = "canned answer"

    resp = await client.post("/eval/llm", json={"prompt": "grade this"})

    assert resp.status_code == 200
    assert resp.json() == {"text": "canned answer"}
    (prompt, cfg), kwargs = mock_call_llm.call_args
    assert prompt == "grade this"
    assert cfg == get_config().ai
    assert kwargs == {"llm_timeout": 120}


@pytest.mark.asyncio
async def test_eval_llm_partial_override_merges_field_level(client, mock_call_llm):
    mock_call_llm.return_value = "ok"

    resp = await client.post(
        "/eval/llm",
        json={"prompt": "p", "llm": {"model": "judge-model"}, "timeout": 300},
    )

    assert resp.status_code == 200
    (_, cfg), kwargs = mock_call_llm.call_args
    # Overridden field applies; omitted fields inherit the backend's values.
    assert cfg.model == "judge-model"
    assert cfg.api_key == get_config().ai.api_key
    assert kwargs["llm_timeout"] == 300


@pytest.mark.asyncio
async def test_eval_llm_invalid_merge_is_422_without_config_leak(client, mock_call_llm):
    resp = await client.post(
        "/eval/llm",
        json={"prompt": "p", "llm": {"max_output_tokens": 999_999_999}},
    )

    assert resp.status_code == 422
    detail = str(resp.json())
    assert "api_key" not in detail
    assert "input_value" not in detail
    mock_call_llm.assert_not_called()


@pytest.mark.asyncio
async def test_eval_llm_provider_failure_is_classified_500(client, mock_call_llm):
    mock_call_llm.side_effect = RuntimeError("boom")

    resp = await client.post("/eval/llm", json={"prompt": "p"})

    assert resp.status_code == 500
    assert resp.json()["detail"] == {"error": "internal_error"}


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, 601])
async def test_eval_llm_timeout_out_of_bounds_is_422(client, mock_call_llm, timeout):
    resp = await client.post("/eval/llm", json={"prompt": "p", "timeout": timeout})

    assert resp.status_code == 422
    mock_call_llm.assert_not_called()
