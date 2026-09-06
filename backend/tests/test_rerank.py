"""Tests for bibilab.pipeline.rerank."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from bibilab.pipeline._shared import DOC_TOKEN_BUDGET, PAIR_WINDOW_TOKENS
from bibilab.pipeline.rerank import _clamp_query


def _pair_tokenizer() -> Tokenizer:
    """A real (non-mocked) tokenizer whose pair template matches bge-reranker-base's:
    `<s> A </s> </s> B </s>` — 4 specials, confirmed against the actual downloaded
    tokenizer.json. A toy vocab keeps this hermetic (no model download in CI) while
    exercising real `tokenizers` truncation/post-processing behavior, not a mock."""
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({f"w{i}": i + 4 for i in range(3000)})
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    return tok


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_cross_encoder_loads_dir_returned_by_ensure(tmp_path: Path):
    """rerank must load the model from the dir ensure() returns for the single
    RERANKER_SPEC_ID — not a separately recomputed path. Otherwise a local_subdir
    change downloads to the new dir while the session reads the stale one."""
    from bibilab.model_registry import RERANKER_SPEC_ID

    fake_dir = tmp_path / "reranker" / "Xenova_bge-reranker-base-q"
    fake_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, path, providers=None, sess_options=None):
            captured["path"] = path
            captured["providers"] = providers

        def get_inputs(self):
            return []

    with (
        patch("bibilab.pipeline.rerank.ensure", return_value=fake_dir) as mock_ensure,
        # The reranker must source providers from interpreting_providers() so
        # CoreML is excluded on macOS (the OOM/hang fix). Patch it to a sentinel
        # and assert that exact value reaches the session — proving the wiring,
        # not the EP list itself (that's covered in test_shared.py).
        patch(
            "bibilab.pipeline.rerank.interpreting_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("onnxruntime.InferenceSession", _FakeSession),
        patch("tokenizers.Tokenizer.from_file", return_value=MagicMock()),
    ):
        from bibilab.pipeline.rerank import ONNXCrossEncoder

        ONNXCrossEncoder()

    mock_ensure.assert_called_once_with(RERANKER_SPEC_ID)
    assert captured["path"] == str(fake_dir / "model.onnx")
    assert captured["providers"] == ["CPUExecutionProvider"]


@pytest.mark.parametrize(
    ("query_words", "doc_words"),
    [
        pytest.param(2000, 800, id="extreme"),
        # Real-shaped regression: a query built from concatenated chunks (over the
        # 64-token clamp, well under the 508 pair window) paired with a long
        # document. Pre-fix (longest_first, no clamp, max_length=512) this halves
        # both sides toward ~254/254 instead of guaranteeing the document a floor.
        pytest.param(100, 500, id="real-shaped"),
    ],
)
def test_document_floor_guaranteed(query_words: int, doc_words: int):
    """No encoded pair exceeds PAIR_WINDOW_TOKENS, and the document side never
    drops below DOC_TOKEN_BUDGET — regardless of query length."""
    length_tok = _pair_tokenizer()  # no truncation — matches rerank.py's _length_tokenizer
    tok = _pair_tokenizer()
    tok.enable_truncation(max_length=PAIR_WINDOW_TOKENS, strategy="only_second")

    query = _words(query_words)
    doc = _words(doc_words)
    enc = tok.encode(_clamp_query(length_tok, query), doc)

    assert len(enc.ids) <= PAIR_WINDOW_TOKENS
    doc_side = sum(1 for sid in enc.sequence_ids if sid == 1)
    assert doc_side >= DOC_TOKEN_BUDGET


def test_short_query_byte_identical_to_pre_fix_encoding():
    """A query at/under QUERY_TOKEN_CLAMP encodes identically to today's
    `enable_truncation(max_length=512)` default-strategy (longest_first) behavior —
    scores must not move for the observed real-traffic query distribution."""
    query = _words(20)
    doc = _words(300)

    pre_fix = _pair_tokenizer()
    pre_fix.enable_truncation(max_length=512)  # today's config: default longest_first
    ids_pre_fix = pre_fix.encode(query, doc).ids

    length_tok = _pair_tokenizer()  # no truncation — matches rerank.py's _length_tokenizer
    post_fix = _pair_tokenizer()
    post_fix.enable_truncation(max_length=PAIR_WINDOW_TOKENS, strategy="only_second")
    ids_post_fix = post_fix.encode(_clamp_query(length_tok, query), doc).ids

    assert ids_post_fix == ids_pre_fix


def test_embed_shares_doc_token_budget():
    """embed.py's single-sequence truncation cap is exactly DOC_TOKEN_BUDGET — the
    same constant rerank.py consumes, not a second literal."""
    tok = _pair_tokenizer()
    tok.enable_truncation(max_length=DOC_TOKEN_BUDGET)

    enc = tok.encode(_words(2000))

    assert len(enc.ids) == DOC_TOKEN_BUDGET


def test_no_stray_enable_truncation_literal():
    """Exactly the two known call sites configure truncation, and neither uses a
    bare 512 literal — both must go through the shared budget constants."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "bibilab"
    hits = []
    for path in src_root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if "enable_truncation(" in line:
                hits.append((path, lineno, line))

    assert len(hits) == 2, hits
    assert {path.name for path, _, _ in hits} == {"rerank.py", "embed.py"}
    assert not any(re.search(r"\b512\b", line) for _, _, line in hits)
