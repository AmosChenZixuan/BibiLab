"""Regression tests for fields removed from BibilabConfig.

Each parametrized case asserts a removed field is absent from a default
BibilabConfig instance, so a future re-introduction is caught.
"""

import pytest
from pydantic import ValidationError

from bibilab.config import BibilabConfig


@pytest.mark.parametrize(
    "path,field",
    [
        ((), "vision"),
        (("accounts", "bilibili"), "last_verified"),
        (("ai",), "transcript_char_limit"),
        (("rag",), "chunk_pause_threshold"),
        ((), "transcript_collection_name"),
        (("transcription",), "llm_timeout"),
        (("backend",), "max_concurrent_downloads"),
        # download_connections is a derived @property, never a stored field —
        # this guards against it being re-added as a configurable knob.
        (("backend",), "download_connections"),
    ],
)
def test_removed_field_absent(path: tuple[str, ...], field: str) -> None:
    cfg = BibilabConfig()
    obj = cfg
    for attr in path:
        obj = getattr(obj, attr)
    assert field not in obj.model_dump()


def test_reranker_spec_id_default_is_quantized_and_resolves() -> None:
    """Quantized is the runtime default (macOS OOM fix); the default must name a
    real registered spec or every reranker download breaks."""
    from bibilab.model_registry import get_spec

    spec_id = BibilabConfig().rag.reranker_spec_id
    assert spec_id == "bge-reranker-base-q"
    assert get_spec(spec_id).kind == "reranker"


def test_reranker_spec_id_rejects_unknown_spec() -> None:
    """The field is a Literal of registered reranker ids, so a config.json typo
    fails loud at load with a field-named ValidationError — not a late KeyError
    deep in required_models / reranker init."""
    with pytest.raises(ValidationError):
        BibilabConfig.model_validate({"rag": {"reranker_spec_id": "bge-reranker-base-int8"}})


def test_backend_download_connections_default() -> None:
    """At the default 4 jobs, the derived per-file aria2c connection count is 16
    (the -x saturation cap), and the total stays within the per-IP budget."""
    cfg = BibilabConfig().backend
    assert cfg.download_connections == 16
    assert cfg.max_concurrent_jobs * cfg.download_connections == 64


def test_backend_download_connections_derived_from_jobs() -> None:
    """download_connections is read-only and scales down with job concurrency so
    jobs × connections never exceeds the 64 per-IP budget; floors at 1."""
    from bibilab.config import BackendConfig

    assert BackendConfig(max_concurrent_jobs=1).download_connections == 16
    assert BackendConfig(max_concurrent_jobs=8).download_connections == 8
    assert BackendConfig(max_concurrent_jobs=128).download_connections == 1
