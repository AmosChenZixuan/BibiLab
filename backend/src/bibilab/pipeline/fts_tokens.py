"""CJK/pinyin tokenization for the FTS5 (BM25) retrieval arm.

Index side (embed.py) and query side (db.query_fts_rows) share these
tokenizers so indexed and queried token shapes always agree. db.py stays
SQL-only; the text processing lives here.
"""

import re

from pypinyin import Style, lazy_pinyin

_CJK_RUN = re.compile(r"[一-鿿㐀-䶿]+")  # BMP + Ext A; Ext B+ (U+20000+) and kana not covered — adequate for zh


def cjk_runs(text: str):
    """Yield (is_cjk: bool, segment: str) pairs, splitting on CJK/non-CJK boundaries."""
    pos = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > pos:
            yield (False, text[pos : m.start()])
        yield (True, m.group(0))
        pos = m.end()
    if pos < len(text):
        yield (False, text[pos:])


def cjk_bigram_tokens(text: str) -> list[str]:
    """Produce unigrams + overlapping bigrams for each CJK run; split non-CJK on whitespace.

    Unigrams ensure single-char CJK words (e.g. 死, 岁, 杀) are searchable.
    Bigrams add adjacency signal so compound words rank above scattered co-occurrence.
    Length-1 CJK runs emit just the unigram. Non-CJK runs are split on whitespace.
    """
    out: list[str] = []
    for is_cjk, seg in cjk_runs(text):
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


def tokenize_cjk(text: str) -> str:
    """Tokenize text for FTS5: unigrams + overlapping bigrams per CJK run.

    Whitespace between adjacent CJK characters is collapsed so character
    bigrams aren't split across it. Non-CJK text is split on whitespace.
    """
    return " ".join(cjk_bigram_tokens(_collapse_cjk_whitespace(text)))


def cjk_query_tokens(text: str) -> list[str]:
    """Produce FTS5 query tokens for a single whitespace-delimited word.

    Multi-char CJK runs emit overlapping bigrams only — no unigrams, to
    avoid AND-term explosion. Single-char CJK runs emit their unigram so
    high-IDF single-character words survive. Non-CJK split on whitespace.
    """
    out: list[str] = []
    for is_cjk, seg in cjk_runs(text):
        if is_cjk:
            if len(seg) == 1:
                out.append(seg)
            else:
                out += [seg[i : i + 2] for i in range(len(seg) - 1)]
        else:
            out += seg.split()
    return out


def pinyin_tokens(text: str) -> list[str]:
    """Toneless-pinyin syllable bigrams for each CJK run.

    Per CJK run: toneless syllables → overlapping bigrams (no unigrams).
    Single-char CJK runs produce nothing — pinyin unigrams are catastrophically
    noisy (~400 toneless syllables). Non-CJK runs are skipped entirely.
    """
    out: list[str] = []
    for is_cjk, seg in cjk_runs(text):
        if not is_cjk:
            continue
        syllables = lazy_pinyin(seg, style=Style.NORMAL)
        if len(syllables) >= 2:
            out += [syllables[i] + syllables[i + 1] for i in range(len(syllables) - 1)]
    return out


def pinyin_index_tokens(text: str) -> str:
    """Toneless-pinyin syllable bigrams, space-joined for FTS5 index column.

    Collapses CJK-CJK whitespace first (same as tokenize_cjk) so pinyin
    bigrams span segment-join gaps — without it the pinyin arm would miss
    homophones straddling a segment boundary that the char arm catches.
    """
    return " ".join(pinyin_tokens(_collapse_cjk_whitespace(text)))


def _fts_quote_token(t: str) -> str:
    """Double-quote and escape a single FTS5 token (disables operator parsing)."""
    return '"' + t.replace('"', '""') + '"'


def escape_fts_query(query_text: str) -> str:
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
    char_tokens = [tok for word in words for tok in cjk_query_tokens(word)]
    py_tokens = [tok for word in words for tok in pinyin_tokens(word)]

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
