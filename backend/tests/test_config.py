"""Regression tests for fields removed from BibilabConfig.

Each parametrized case asserts a removed field is absent from a default
BibilabConfig instance, so a future re-introduction is caught.
"""

import pytest

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
        # reranker_spec_id (#567) removed in #573 — one int8 reranker ships, the
        # spec is a model_registry constant, not a config knob.
        (("rag",), "reranker_spec_id"),
    ],
)
def test_removed_field_absent(path: tuple[str, ...], field: str) -> None:
    cfg = BibilabConfig()
    obj = cfg
    for attr in path:
        obj = getattr(obj, attr)
    assert field not in obj.model_dump()


def test_legacy_reranker_spec_id_in_config_is_ignored() -> None:
    """An existing config.json written by #567 carries rag.reranker_spec_id. After
    #573 dropped the field, loading such a file must NOT brick (Pydantic
    extra='ignore') and the stale key must not survive a round-trip — otherwise an
    upgrade either 422s on load or silently re-persists a dead knob."""
    cfg = BibilabConfig.model_validate({"rag": {"reranker_spec_id": "bge-reranker-base"}})
    assert "reranker_spec_id" not in cfg.rag.model_dump()


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
