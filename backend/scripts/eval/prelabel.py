"""LLM-assisted pre-labeling for #220 eval chunks.

Reads labels.json, calls the LLM to judge (query, chunk) pairs in batches,
writes back with 'relevant' pre-filled.

Usage:
    uv run python scripts/eval/prelabel.py                    # default sweep-001
    uv run python scripts/eval/prelabel.py --run-id sweep-001
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from bibilab.config import get_config
from bibilab.pipeline._shared import _call_llm

BATCH_SIZE = 5

BATCH_PROMPT = """You are an evaluator judging whether transcript excerpts are relevant to a user's question.

Question: {query}

For each excerpt below, answer "yes" if it contains information that helps answer the question, "no" otherwise.
- "yes" = excerpt directly addresses what the question asks, even partially.
- "no" = off-topic, wrong video, or doesn't contain the specific information asked.
- Meta-questions about "how many" that require counting across sources: always "no"
  (a single excerpt can never contain the list-level count).

Return ONLY a JSON array of "yes"/"no" strings, one per excerpt. No explanation.

{excerpts}"""

# Matches yes/no regardless of surrounding formatting: "Yes.", "**no**", '"yes"', etc.
_YES_RE = re.compile(r"(?<![a-zA-Z])yes(?![a-zA-Z])", re.IGNORECASE)


def _parse_yes_no(text: str) -> bool:
    return bool(_YES_RE.search(text))


def _judge_batch_sync(query: str, chunks: list[dict], ai_cfg) -> list[bool]:
    excerpts = "\n".join(
        f'[{i}] [{c["video_title"]} @ {c["timestamp_start"]}s-{c["timestamp_end"]}s]: "{c["content"][:1200]}"'
        for i, c in enumerate(chunks)
    )
    prompt = BATCH_PROMPT.format(query=query, excerpts=excerpts)
    try:
        response = _call_llm(prompt=prompt, cfg=ai_cfg, llm_max_tokens=2048, llm_timeout=120)
    except Exception as exc:
        print(f"  LLM error: {exc}")
        return [False] * len(chunks)

    # Try to extract JSON array from response
    text = response.strip()
    try:
        results = json.loads(text)
        if isinstance(results, list) and len(results) == len(chunks):
            return [_parse_yes_no(str(r)) for r in results]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: try to find JSON array anywhere in response
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            results = json.loads(match.group())
            if isinstance(results, list) and len(results) == len(chunks):
                return [_parse_yes_no(str(r)) for r in results]
        except (json.JSONDecodeError, TypeError):
            pass

    print(f"  Parse warning: expected JSON array, got: {text[:120]}")
    return [False] * len(chunks)


async def main():
    parser = argparse.ArgumentParser(description="LLM pre-label #220 eval chunks")
    parser.add_argument("--run-id", default="sweep-001", help="Sweep run subdirectory")
    args = parser.parse_args()

    labels_path = Path.home() / ".bibilab" / "eval" / args.run_id / "labels.json"
    data = json.loads(labels_path.read_text(encoding="utf-8"))

    cfg = get_config()
    ai_cfg = cfg.ai
    query_ids = sorted(k for k in data if k != "combos")

    yes_count = 0
    total = 0
    processed = 0

    for qid in query_ids:
        query_text = data[qid]["query_text"]
        chunks = data[qid]["chunks"]

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]
            results = await asyncio.to_thread(_judge_batch_sync, query_text, batch, ai_cfg)

            for i, relevant in enumerate(results):
                total += 1
                processed += 1
                chunk = batch[i]
                chunk["relevant"] = relevant
                if relevant:
                    yes_count += 1

            markers = " ".join("Y" if r else "n" for r in results)
            print(
                f"[{processed}/{sum(len(data[q]['chunks']) for q in query_ids)}] "
                f"{qid} batch {batch_start // BATCH_SIZE + 1}: [{markers}]  "
                f"({yes_count}/{total} yes)"
            )

    labels_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    pct = yes_count / total * 100 if total else 0
    print(f"\nPre-labeling complete. {yes_count}/{total} chunks marked relevant ({pct:.1f}%)")
    print(f"Review in: {labels_path}")
    print("Focus on verifying the YES chunks (false positives) and spot-check ~10% of NO chunks.")


if __name__ == "__main__":
    asyncio.run(main())
