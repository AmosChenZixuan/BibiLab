"""FTS5 helpers + CJK/pinyin tokenization."""

from __future__ import annotations

import logging
import re

import aiosqlite
from pypinyin import Style, lazy_pinyin

from bibilab.db.connection import get_db
from bibilab.db.sources import _in_placeholders

logger = logging.getLogger(__name__)


async def clear_fts_for_list(list_id: str) -> None:
    """Delete all FTS rows whose source_id belongs to sources in the given list."""
    async with get_db() as db:
        await db.execute(
            "DELETE FROM chunks_fts WHERE source_id IN (SELECT id FROM sources WHERE list_id = ?)",
            (list_id,),
        )
        await db.commit()


_CJK = re.compile(r"[一-鿿㐀-䶿]")  # BMP + Ext A; Ext B+ (U+20000+) and kana not covered — adequate for zh

_CJK_RUN = re.compile(r"[一-鿿㐀-䶿]+")


def _cjk_runs(text: str):
    """Yield (is_cjk: bool, segment: str) pairs, splitting on CJK/non-CJK boundaries."""
    pos = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > pos:
            yield (False, text[pos : m.start()])
        yield (True, m.group(0))
        pos = m.end()
    if pos < len(text):
        yield (False, text[pos:])


def _cjk_bigram_tokens(text: str) -> list[str]:
    """Produce unigrams + overlapping bigrams for each CJK run; split non-CJK on whitespace.

    Unigrams ensure single-char CJK words (e.g. 死, 岁, 杀) are searchable.
    Bigrams add adjacency signal so compound words rank above scattered co-occurrence.
    Length-1 CJK runs emit just the unigram. Non-CJK runs are split on whitespace.
    """
    out: list[str] = []
    for is_cjk, seg in _cjk_runs(text):
        if is_cjk:
            out += list(seg)  # unigrams
            out += [seg[i : i + 2] for i in range(len(seg) - 1)]  # bigrams
        else:
            out += seg.split()
    return out


def _collapse_cjk_whitespace(text: str) -> str:
    """Remove whitespace between adjacent CJK characters.

    chunk.py joins transcript segments with spaces; collapsing CJK-CJK gaps
    lets bigrams span them. Shared by the char and pinyin index tokenizers so
    both span segment joins identically.
    """
    return re.sub(r"(?<=[一-鿿㐀-䶿])\s+(?=[一-鿿㐀-䶿])", "", text)


def _tokenize_cjk(text: str) -> str:
    """Tokenize text for FTS5: unigrams + overlapping bigrams per CJK run.

    Whitespace between adjacent CJK characters is collapsed so character
    bigrams aren't split across it. Non-CJK text is split on whitespace.
    """
    return " ".join(_cjk_bigram_tokens(_collapse_cjk_whitespace(text)))


def _cjk_query_tokens(text: str) -> list[str]:
    """Produce FTS5 query tokens for a single whitespace-delimited word.

    Multi-char CJK runs emit overlapping bigrams only — no unigrams, to
    avoid AND-term explosion. Single-char CJK runs emit their unigram so
    high-IDF single-character words survive. Non-CJK split on whitespace.
    """
    out: list[str] = []
    for is_cjk, seg in _cjk_runs(text):
        if is_cjk:
            if len(seg) == 1:
                out.append(seg)
            else:
                out += [seg[i : i + 2] for i in range(len(seg) - 1)]
        else:
            out += seg.split()
    return out


def _pinyin_tokens(text: str) -> list[str]:
    """Toneless-pinyin syllable bigrams for each CJK run.

    Per CJK run: toneless syllables → overlapping bigrams (no unigrams).
    Single-char CJK runs produce nothing — pinyin unigrams are catastrophically
    noisy (~400 toneless syllables). Non-CJK runs are skipped entirely.
    """
    out: list[str] = []
    for is_cjk, seg in _cjk_runs(text):
        if not is_cjk:
            continue
        syllables = lazy_pinyin(seg, style=Style.NORMAL)
        if len(syllables) >= 2:
            out += [syllables[i] + syllables[i + 1] for i in range(len(syllables) - 1)]
    return out


def _pinyin_index_tokens(text: str) -> str:
    """Toneless-pinyin syllable bigrams, space-joined for FTS5 index column.

    Collapses CJK-CJK whitespace first (same as _tokenize_cjk) so pinyin
    bigrams span segment-join gaps — without it the pinyin arm would miss
    homophones straddling a segment boundary that the char arm catches.
    """
    return " ".join(_pinyin_tokens(_collapse_cjk_whitespace(text)))


def _fts_quote_token(t: str) -> str:
    """Double-quote and escape a single FTS5 token (disables operator parsing)."""
    return '"' + t.replace('"', '""') + '"'


def _escape_fts_query(query_text: str) -> str:
    """Escape user input for safe FTS5 MATCH evaluation.

    Returns a column-scoped expression targeting the content and/or pinyin
    FTS5 columns so the BM25 arm survives homophone ASR errors (e.g. query
    '苹果' matches indexed '平果' via shared toneless-pinyin bigrams):

        content : ("a" OR "b") OR pinyin : ("x" OR "y")

    Tokens within an arm are OR-joined, not AND-joined: a multi-syllable term or
    a natural-language question ('苹果是什么') is one CJK run whose overlapping
    bigrams must not all be required to co-occur — under AND a single divergent
    syllable (or an interrogative like '是谁') zeroes the arm, so recall collapses
    for anything past a two-character word. OR lets BM25 rank by matched
    bigrams; high-IDF entity bigrams dominate and stop-word bigrams sit in the
    truncated tail. Each arm parenthesizes its token list so the column filter
    binds to the whole group — FTS5's `col :` prefix otherwise scopes only the
    first phrase, leaving later tokens to probe every column. With the parens,
    char tokens never probe pinyin and pinyin tokens never probe content. When
    the pinyin arm produces no tokens (English, single-char CJK, no CJK at all)
    the OR is omitted and only the content arm is returned.
    """
    words = query_text.split()
    char_tokens = [tok for word in words for tok in _cjk_query_tokens(word)]
    py_tokens = [tok for word in words for tok in _pinyin_tokens(word)]

    if not char_tokens and not py_tokens:
        return ""

    char_arm = ""
    if char_tokens:
        char_arm = "content : (" + " OR ".join(_fts_quote_token(t) for t in char_tokens) + ")"

    py_arm = ""
    if py_tokens:
        py_arm = "pinyin : (" + " OR ".join(_fts_quote_token(t) for t in py_tokens) + ")"

    if char_arm and py_arm:
        return f"{char_arm} OR {py_arm}"
    return char_arm or py_arm


async def query_fts_rows(
    query_text: str,
    source_ids: list[str],
    top_k: int = 30,
) -> list[aiosqlite.Row]:
    """Run FTS5 MATCH query filtered by source_ids, return rows ranked by BM25.

    Returns an empty list if the query is empty or FTS5 raises a syntax error.
    """
    if not source_ids:
        return []
    match_query = _escape_fts_query(query_text)
    if not match_query:
        return []
    placeholders = _in_placeholders(source_ids)
    async with get_db() as db:
        try:
            cursor = await db.execute(
                f"SELECT content, source_id, video_title, timestamp_start, timestamp_end, rank, chunk_id, "
                f"seg_start, seg_end "
                f"FROM chunks_fts "
                f"WHERE chunks_fts MATCH ? AND source_id IN ({placeholders}) "
                f"ORDER BY rank "
                f"LIMIT ?",
                [match_query, *source_ids, top_k],
            )
            return await cursor.fetchall()
        except aiosqlite.OperationalError as exc:
            logger.warning("FTS5 MATCH query failed (%s); returning empty results", exc)
            return []
