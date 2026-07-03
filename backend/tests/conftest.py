from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from bibilab.pipeline._shared import StreamEvent
from tests import an_async_generator


@pytest.fixture(autouse=True)
def _clear_llm_client_caches():
    """Prevent MagicMock accumulation in module-level client caches across tests."""
    yield
    from bibilab.pipeline._shared import _async_client_cache, _client_cache

    _async_client_cache.clear()
    _client_cache.clear()


async def _default_stream_llm(*args, **kwargs):
    """Default no-op stream: yield a single `done` event so the LLM-shaped code path
    accepts the empty stream and the consumer sees a clean terminal event."""
    async for ev in an_async_generator([StreamEvent(type="done")]):
        yield ev


_DEFAULT_LLM_RESPONSE = "{}"


@pytest.fixture()
def mock_stream_llm():
    """Patch the canonical `bibilab.routers.chat.stream_llm` seam with a configurable fake.

    Default behavior: yields a single `StreamEvent(type="done")` (no-op stream).

    Override per test by reassigning attributes on the yielded mock:
        mock_stream_llm.side_effect = my_async_gen_fn      # custom async generator
        mock_stream_llm.side_effect = MyException()       # raise on call
        mock_stream_llm.side_effect = [ev1, ev2, ev3]      # sequence of return values
    """
    mock = MagicMock(side_effect=_default_stream_llm)
    with patch("bibilab.routers.chat.stream_llm", mock):
        yield mock


@pytest.fixture()
def mock_call_llm():
    """Patch every `_call_llm` reference in the codebase with one fake.

    The unified `_call_llm_with_retry` in `pipeline/_shared.py` is the
    canonical call path, but digest, chat_summary, and worker each `from
    bibilab.pipeline._shared import _call_llm` at module level — that
    re-import binds a *separate* name, and patching the source module's
    attribute doesn't reach it. Patch the source AND each re-import site
    so the mock is effective everywhere.

    Default return: "{}" (empty JSON object — the lowest common denominator
    across digest/chat_summary/artifact LLM consumers, all of which parse JSON).

    Override per test by reassigning attributes on the yielded mock:
        mock_call_llm.return_value = "..."                 # canned string
        mock_call_llm.side_effect = my_fn                  # custom callable
        mock_call_llm.side_effect = MyException()          # raise on call
        mock_call_llm.side_effect = [err, valid, valid]    # sequence (e.g. retries)
    """
    mock = MagicMock(return_value=_DEFAULT_LLM_RESPONSE)
    with (
        patch("bibilab.pipeline._shared._call_llm", mock),
        # chat_summary still re-imports _call_llm at module scope — that
        # re-import binds a separate name, so patching the source module
        # alone doesn't reach it. Patch the re-import site too.
        patch("bibilab.pipeline.chat_summary._call_llm", mock),
        # routers/eval.py (POST /eval/llm) re-imports it the same way.
        patch("bibilab.routers.eval._call_llm", mock),
    ):
        yield mock


@pytest.fixture(autouse=True)
def _mock_model_gate():
    """Prevent pre-flight 412 in tests — models are never downloaded in CI."""
    with patch("bibilab.routers._model_gate.missing_required_models", return_value=[]):
        yield


class _MockEmbeddingFunction:
    def __call__(self, input):
        return [[0.0] * 384 for _ in input]

    def name(self):
        return "mock_embedding"


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from bibilab.config import _reset_cache

    _reset_cache()
    mock_ef = _MockEmbeddingFunction()
    monkeypatch.setenv("BIBILAB_HOME", str(tmp_path))
    with patch("bibilab.pipeline.embed._default_embedding_function", return_value=mock_ef):
        yield tmp_path


@pytest.fixture()
def downloads_dir(tmp_bibilab_home: Path) -> Path:
    """The ~/.bibilab/downloads directory, created and ready for temp files."""
    path = tmp_bibilab_home / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest_asyncio.fixture()
async def client(tmp_bibilab_home: Path):  # noqa: ARG001
    from bibilab.main import create_app

    with patch("bibilab.main.WEB_DIST", tmp_bibilab_home / "web-dist-disabled"):
        app = create_app(start_worker=False)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as async_client:
                yield async_client
                from bibilab.pipeline.chat_runs import ChatRunRegistry, get_chat_run_registry

                old = get_chat_run_registry()
                for _, task in old.all_tasks():
                    if not task.done():
                        task.cancel()
                for task in old.all_background_tasks():
                    if not task.done():
                        task.cancel()
                # Replace with fresh registry for next test
                import bibilab.pipeline.chat_runs as cr

                cr._registry = ChatRunRegistry()

                from bibilab.pipeline._shared import _async_client_cache, _client_cache

                _async_client_cache.clear()
                _client_cache.clear()
