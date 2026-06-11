from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bibilab.db import get_sections, get_segments_for_ranges
from bibilab.pipeline._shared import _call_llm

from eval._utils import strip_json_fences

EXTRACTION_LLM_TIMEOUT = 180

_EXTRACT_PROMPT = """你是一个信息抽取助手。下面是一段视频文字稿。请抽取其中**明确陈述**的原子事实。

强约束：
- 每条事实一句话，只用文字稿里出现的信息，不要推断、补充或联想。
- entities：该事实里出现的专有名词（人名、组织、地名、设定名）；没有就空数组。
- is_cause：这条事实是否在**解释某件事为什么发生 / 动机 / 由来**（是→true）。
- has_time：这条事实是否带**时间或先后顺序**标记（"之后""第二天""先……再……"等，是→true）。
- 文字稿空白或纯非语言标记时，返回空数组。

只返回单个 JSON 对象，无 markdown fences，无多余文字：
{"claims": [{"text": "...", "entities": ["..."], "is_cause": false, "has_time": false}]}"""

_LANG_INSTRUCTION = {
    "en": "\n\nIMPORTANT: write every `text` field in English.",
    "zh": "",
}


def _safe_call(prompt: str, ai_cfg: Any) -> tuple[str | None, str | None]:
    try:
        return (_call_llm(prompt, ai_cfg, llm_timeout=EXTRACTION_LLM_TIMEOUT), None)
    except Exception as e:  # timeout/API error on one span must not crash the pool
        return (None, f"{type(e).__name__}: {e}")


def extract_claims_for_span(span: dict, ai_cfg: Any, language: str = "zh") -> tuple[list[Claim], str | None]:
    prompt = f"{_EXTRACT_PROMPT}{_LANG_INSTRUCTION.get(language, '')}\n\n文字稿：\n{span['text']}"
    raw, call_err = _safe_call(prompt, ai_cfg)
    if raw is None:
        return ([], call_err)
    try:
        data = json.loads(strip_json_fences(raw))
        items = data.get("claims", []) if isinstance(data, dict) else []
    except json.JSONDecodeError as e:
        return ([], f"JSONDecodeError: {e}")
    snippet = span["text"][:120]
    claims = [
        Claim(
            source_id=span["source_id"],
            section_seq=span["section_seq"],
            text=str(it.get("text", "")).strip(),
            snippet=snippet,
            entities=[str(x) for x in it.get("entities", []) if x],
            is_cause=bool(it.get("is_cause", False)),
            has_time=bool(it.get("has_time", False)),
        )
        for it in items
        if str(it.get("text", "")).strip()
    ]
    return (claims, None)


class Claim(BaseModel):
    source_id: str
    section_seq: int
    text: str
    snippet: str = ""
    entities: list[str] = Field(default_factory=list)
    is_cause: bool = False   # explains WHY some event happened
    has_time: bool = False   # carries an explicit order / time marker


async def load_spans(source_id: str) -> list[dict]:
    """One span per section: {source_id, section_seq, text}. Reads each section's
    segments once via the shared range helper."""
    sections = await get_sections(source_id)
    spans: list[dict] = []
    for sec in sections:
        rows = await get_segments_for_ranges([(source_id, sec["seg_start"], sec["seg_end"])])
        text = " ".join(r["text"] for r in rows)
        spans.append({"source_id": source_id, "section_seq": sec["seq"], "text": text})
    return spans


def _cache_dir() -> Path:
    from bibilab.config import bibilab_home
    d = bibilab_home() / "evals" / "_claims_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(span: dict) -> Path:
    h = hashlib.sha1(span["text"].encode("utf-8")).hexdigest()[:16]
    return _cache_dir() / f"{span['source_id']}_{span['section_seq']}_{h}.json"


def build_claim_pool(spans: list[dict], ai_cfg: Any, language: str = "zh") -> tuple[list[Claim], list[str]]:
    pool: list[Claim] = []
    errors: list[str] = []

    def _one(span: dict) -> tuple[list[Claim], str | None]:
        cp = _cache_path(span)
        if cp.exists():
            return ([Claim.model_validate(c) for c in json.loads(cp.read_text())], None)
        claims, err = extract_claims_for_span(span, ai_cfg, language)
        if err is None:
            cp.write_text(json.dumps([c.model_dump() for c in claims], ensure_ascii=False))
        return (claims, err)

    if not spans:
        return ([], [])
    with ThreadPoolExecutor(max_workers=min(len(spans), 8)) as pool_exec:
        futures = {pool_exec.submit(_one, s): s for s in spans}
        for fut in as_completed(futures):
            s = futures[fut]
            claims, err = fut.result()
            pool.extend(claims)
            if err is not None:
                errors.append(f"source {s['source_id']} sec {s['section_seq']}: {err}")
    return (pool, errors)
