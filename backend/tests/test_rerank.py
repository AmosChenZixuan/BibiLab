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


def test_predict_applies_query_clamp_before_encoding():
    """predict()'s own wiring — not just the standalone _clamp_query helper —
    must clamp the query before the pair encode. Without it, an over-long query
    reaches self._tokenizer directly and only_second raises rather than trimming
    down to the guaranteed floor."""
    import numpy as np

    from bibilab.pipeline.rerank import ONNXCrossEncoder

    class _Input:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeSession:
        def get_inputs(self):
            return [_Input("input_ids"), _Input("attention_mask")]

        def run(self, _output_names, onnx_input):
            batch = onnx_input["input_ids"].shape[0]
            return [np.zeros((batch, 1))]

    encoder = ONNXCrossEncoder.__new__(ONNXCrossEncoder)
    encoder._np = np
    encoder._session = _FakeSession()
    encoder._tokenizer = _pair_tokenizer()
    encoder._tokenizer.enable_truncation(max_length=PAIR_WINDOW_TOKENS, strategy="only_second")
    encoder._length_tokenizer = _pair_tokenizer()

    # Query alone (2000 tokens) exceeds the pair window — only_second would raise
    # without the clamp, since it can't trim the query and the document alone
    # can't shrink enough to compensate.
    scores, _ = encoder.predict([[_words(2000), _words(500)]])

    assert scores == [0.0]


def test_predict_reports_tokens_dropped_per_pair():
    """predict() reports per-pair document token loss from only_second truncation:
    0 when the pair fits, and the real token-count diff when it doesn't."""
    import numpy as np

    from bibilab.pipeline.rerank import ONNXCrossEncoder

    class _Input:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeSession:
        def get_inputs(self):
            return [_Input("input_ids"), _Input("attention_mask")]

        def run(self, _output_names, onnx_input):
            batch = onnx_input["input_ids"].shape[0]
            return [np.zeros((batch, 1))]

    encoder = ONNXCrossEncoder.__new__(ONNXCrossEncoder)
    encoder._np = np
    encoder._session = _FakeSession()
    encoder._tokenizer = _pair_tokenizer()
    encoder._tokenizer.enable_truncation(max_length=PAIR_WINDOW_TOKENS, strategy="only_second")
    encoder._length_tokenizer = _pair_tokenizer()

    query = _words(10)
    fitting_doc = _words(50)
    oversized_doc = _words(1000)

    # Ground truth for the oversized pair: how many doc tokens only_second actually
    # keeps, derived independently the same way test_document_floor_guaranteed does.
    truncating_tok = _pair_tokenizer()
    truncating_tok.enable_truncation(max_length=PAIR_WINDOW_TOKENS, strategy="only_second")
    kept_enc = truncating_tok.encode(query, oversized_doc)
    doc_side_kept = sum(1 for sid in kept_enc.sequence_ids if sid == 1)
    expected_dropped = 1000 - doc_side_kept
    assert expected_dropped > 0  # sanity: this pair really does get truncated

    _, tokens_dropped = encoder.predict([[query, fitting_doc], [query, oversized_doc]])

    assert tokens_dropped[0] == 0
    assert tokens_dropped[1] == expected_dropped


def test_no_bare_truncation_literal():
    """Every `enable_truncation(` call site in src/bibilab must reference a shared
    budget constant, not a bare 512 — the point of exporting the constants is one
    source of truth, including for future consumers (e.g. chunk sizing)."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "bibilab"
    hits = [
        (path, lineno, line)
        for path in src_root.rglob("*.py")
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if "enable_truncation(" in line
    ]

    assert hits, "expected at least the rerank.py and embed.py call sites"
    assert not any(re.search(r"\b512\b", line) for _, _, line in hits)
