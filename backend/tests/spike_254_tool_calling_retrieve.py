"""Spike: re-run #254 tool-calling test with tool renamed "retrieve" instead of "search".

Throwaway script — delete after decision.
"""

import asyncio
import json
import sys

sys.path.insert(0, "src")

from bibilab.config import AIConfig
from bibilab.pipeline._shared import ToolDefinition, stream_llm

RETRIEVE_TOOL = ToolDefinition(
    name="retrieve",
    description=(
        "Retrieve information from video transcripts. Use when the user asks about "
        "video content, facts, comparisons, or summaries. Do NOT use for chitchat "
        "(thanks, greetings, rephrasing) or conversation-only queries."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — key terms or question in the user's language",
            },
            "search_mode": {
                "type": "string",
                "enum": ["factual", "breadth", "analytical"],
                "description": (
                    "factual = specific fact from 1-2 sources; "
                    "breadth = survey/list across many sources; "
                    "analytical = comparison/analysis needing deep per-source coverage"
                ),
            },
        },
        "required": ["query", "search_mode"],
    },
)

SYSTEM_PROMPT = (
    "You have a tool called `retrieve` that searches video transcript content.\n"
    "Use it when the user asks about facts, comparisons, or summaries from videos.\n"
    "Skip it for chitchat, greetings, or conversation-only questions."
)

QUERIES = [
    ("F1", "什么时候提到了注意力机制？", "tool"),
    ("F2", "when did the host say transformer architecture", "tool"),
    ("F3", "舒芙蕾需要几个鸡蛋？", "tool"),
    ("B1", "compare what each video says about model evaluation", "tool"),
    ("B2", "汇总所有视频的核心观点", "tool"),
    ("C1", "thanks", "skip"),
    ("C2", "rephrase that", "skip"),
    ("C3", "what did you just say", "skip"),
    ("M1", "how many videos are in this list", "n/a"),
    ("M2", "which video is the longest", "n/a"),
]

MODELS = ["qwen3:8b", "glm-4.7-flash"]


async def test_model(model: str) -> list[dict]:
    cfg = AIConfig(
        protocol="openai",
        model=model,
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )
    results = []
    for qid, query, expect in QUERIES:
        try:
            tool_calls = []
            async for event in stream_llm(
                messages=[{"role": "user", "content": query}],
                cfg=cfg,
                tools=[RETRIEVE_TOOL],
                system=SYSTEM_PROMPT,
                llm_max_tokens=512,
            ):
                if event.type == "tool_call":
                    tool_calls.append(event.tool_call)
            if tool_calls:
                tc = tool_calls[0]
                results.append(
                    {
                        "id": qid,
                        "query": query,
                        "expect": expect,
                        "called": True,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "verdict": "✓" if expect == "tool" else "✗ unexpected call",
                    }
                )
            else:
                results.append(
                    {
                        "id": qid,
                        "query": query,
                        "expect": expect,
                        "called": False,
                        "verdict": "✓" if expect in ("skip", "n/a") else "✗ missed call",
                    }
                )
        except Exception as exc:
            results.append(
                {
                    "id": qid,
                    "query": query,
                    "expect": expect,
                    "called": False,
                    "verdict": f"✗ error: {exc}",
                }
            )
    return results


async def main():
    for model in MODELS:
        print(f"\n=== {model} ===")
        results = await test_model(model)
        correct = 0
        for r in results:
            print(f"  {r['id']}: {r['verdict']}", end="")
            if r.get("name"):
                print(f"  → {r['name']}({json.dumps(r['arguments'], ensure_ascii=False)})", end="")
            print()
            if "✓" in r["verdict"]:
                correct += 1
        print(f"  Score: {correct}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
