import { describe, expect, test } from "vitest";

import { FIND_PASSAGES_TOOL_NAME } from "@/lib/utils";
import {
  autoResize,
  buildMindmapAskMessage,
  coerceCitationEvent,
  coerceContentBlock,
  facetNoMatchHint,
  formatDurationHuman,
  formatSubtitle,
  formatTimestamp,
  stripLegacyTokens,
  type RetrievalCall,
} from "@/lib/chat-utils";

describe("formatDurationHuman", () => {
  test("seconds only", () => {
    expect(formatDurationHuman(45)).toBe("45s");
  });

  test("minutes only", () => {
    expect(formatDurationHuman(300)).toBe("5m");
  });

  test("hours and minutes", () => {
    expect(formatDurationHuman(3720)).toBe("1h 2m");
  });

  test("exact hours", () => {
    expect(formatDurationHuman(7200)).toBe("2h");
  });

  test("zero", () => {
    expect(formatDurationHuman(0)).toBe("0s");
  });
});

describe("formatSubtitle", () => {
  const t = (key: string, params?: Record<string, string | number>) => {
    const map: Record<string, string> = {
      "chat.subtitle.templateSingular": "%{count} source · %{duration} total",
      "chat.subtitle.templatePlural": "%{count} sources · %{duration} total",
    };
    const value = map[key] ?? key;
    if (!params) return value;
    return value.replace(/%\{(\w+)\}/g, (_, k) => String(params[k]));
  };

  test("single source", () => {
    expect(formatSubtitle(t, 1, 120)).toBe("1 source · 2m total");
  });

  test("multiple sources", () => {
    expect(formatSubtitle(t, 3, 3600)).toBe("3 sources · 1h total");
  });
});

describe("stripLegacyTokens", () => {
  test("removes [Title @ Ns-Ns] patterns", () => {
    expect(stripLegacyTokens("See [My Video @ 120s-145s].")).toBe("See .");
  });

  test("removes multiple", () => {
    expect(stripLegacyTokens("[A @ 10s-20s] and [B @ 30s-40s]")).toBe(" and ");
  });

  test("leaves non-citation brackets intact", () => {
    expect(stripLegacyTokens("array[0] and [note]")).toBe("array[0] and [note]");
  });

  test("handles empty string", () => {
    expect(stripLegacyTokens("")).toBe("");
  });
});

describe("formatTimestamp", () => {
  test("formats ISO string to time", () => {
    const result = formatTimestamp("2026-04-24T14:30:00Z");
    expect(result).toMatch(/\d{1,2}:\d{2}/);
  });
});

describe("autoResize", () => {
  test("resets height when empty", () => {
    const ta = { value: "", style: { height: "100px", overflowY: "auto" }, scrollHeight: 0 } as unknown as HTMLTextAreaElement;
    autoResize(ta);
    expect(ta.style.height).toBe("auto");
    expect(ta.style.overflowY).toBe("hidden");
  });

  test("sets height to scrollHeight when under max", () => {
    const ta = { value: "text", style: { height: "", overflowY: "" }, scrollHeight: 80 } as unknown as HTMLTextAreaElement;
    autoResize(ta);
    expect(ta.style.height).toBe("80px");
    expect(ta.style.overflowY).toBe("hidden");
  });

  test("caps at 200px and shows scrollbar", () => {
    const ta = { value: "text", style: { height: "", overflowY: "" }, scrollHeight: 300 } as unknown as HTMLTextAreaElement;
    autoResize(ta);
    expect(ta.style.height).toBe("200px");
    expect(ta.style.overflowY).toBe("auto");
  });
});

describe("RetrievalCall", () => {
  test("has all required fields (v2 find_passages)", () => {
    const call: RetrievalCall = {
      query: "test query",
      tool_name: FIND_PASSAGES_TOOL_NAME,
      candidates_evaluated: 5,
      sources_with_hits: 2,
      sources_total: 3,
      section_coverage: [],
      context: [
        {
          chunk_id: "c1",
          citation_index: 1,
          section_id: "sec-1",
          section_seq: 1,
          source_id: "s1",
          source_title: "Test Video",
          timestamp_start: 120.4,
          timestamp_end: 145.0,
          rerank_score: 0.95,
          preview: "test content here",
        },
      ],
      reranked: true,
      scoped_pool_size: 3,
    };
    expect(call.tool_name).toBe(FIND_PASSAGES_TOOL_NAME);
    expect(call.context![0].chunk_id).toBe("c1");
    expect(call.context![0].citation_index).toBe(1);
    expect(call.context![0].timestamp_start).toBe(120.4);
    expect(call.context![0].rerank_score).toBe(0.95);
  });

  test("context can be empty array", () => {
    const call: RetrievalCall = {
      query: "test",
      tool_name: FIND_PASSAGES_TOOL_NAME,
      candidates_evaluated: 0,
      sources_with_hits: 0,
      sources_total: 1,
      section_coverage: [],
      context: [],
      reranked: false,
      scoped_pool_size: 1,
    };
    expect(call.context).toHaveLength(0);
  });
});

describe("facetNoMatchHint", () => {
  const t = (key: string, params?: Record<string, string | number>) => {
    const map: Record<string, string> = {
      "chat.ledger.facet.sequence": "#%{n}",
      "chat.ledger.facet.season": "season %{n}",
      "chat.ledger.facetNoMatch": "No source matched %{facets} — searched all sources instead.",
      "chat.ledger.facetNoMatchGeneric": "No source matched the requested filter — searched all sources instead.",
    };
    const value = map[key] ?? key;
    if (!params) return value;
    return value.replace(/%\{(\w+)\}/g, (_, k) => String(params[k]));
  };

  test("sequence only", () => {
    expect(facetNoMatchHint(t, { sequence_number: 8, season_number: null, matched_count: 0, no_match: true }))
      .toBe("No source matched #8 — searched all sources instead.");
  });

  test("season only", () => {
    expect(facetNoMatchHint(t, { sequence_number: null, season_number: 2, matched_count: 0, no_match: true }))
      .toBe("No source matched season 2 — searched all sources instead.");
  });

  test("both facets joined", () => {
    expect(facetNoMatchHint(t, { sequence_number: 8, season_number: 2, matched_count: 0, no_match: true }))
      .toBe("No source matched #8, season 2 — searched all sources instead.");
  });

  test("no facets falls back to generic", () => {
    expect(facetNoMatchHint(t, { sequence_number: null, season_number: null, matched_count: null, no_match: true }))
      .toBe("No source matched the requested filter — searched all sources instead.");
  });
});

describe("coerceCitationEvent", () => {
  // The SSE citation event comes over the wire with section_id as a
  // JSON number (backend's CitationRegistryEntry.section_id is the
  // INTEGER sections.id). The FE's ContentBlock.citation declares
  // section_id: string, and SourcesViewerMode.resolveTargetIdx does
  // strict equality against SourceSection.section_id (also string) —
  // a number on one side silently falls through to the timestampStart
  // branch, landing the reader on the wrong section. The coercion
  // normalizes at the SSE-consumer boundary so the type contract and
  // the jump both work.

  test("coerces section_id from number to string", () => {
    const result = coerceCitationEvent({
      type: "citation",
      index: 3,
      section_id: 7,                       // number, as the wire delivers
      source_id: "src-1",
      timestamp_start: 42.5,
      chunk_ids: ["c1", "c2"],
    });
    expect(result).toEqual({
      type: "citation",
      index: 3,
      section_id: "7",
      source_id: "src-1",
      timestamp_start: 42.5,
      chunk_ids: ["c1", "c2"],
    });
  });

  test("passes through a string section_id unchanged", () => {
    const result = coerceCitationEvent({
      type: "citation",
      index: 1,
      section_id: "sec-1",
      source_id: "src-1",
      timestamp_start: 42,
      chunk_ids: ["c1"],
    });
    if (result.type !== "citation") throw new Error("expected citation");
    expect(result.section_id).toBe("sec-1");
  });

  test("defaults to empty section_id when missing (so caller's falsy check skips the sectionId branch)", () => {
    const result = coerceCitationEvent({
      type: "citation",
      index: 1,
      source_id: "src-1",
      timestamp_start: 42,
      chunk_ids: [],
    });
    if (result.type !== "citation") throw new Error("expected citation");
    expect(result.section_id).toBe("");
  });

  test("normalizes chunk_ids to an array of strings", () => {
    const result = coerceCitationEvent({
      type: "citation",
      index: 1,
      section_id: "1",
      source_id: "src-1",
      timestamp_start: 0,
      chunk_ids: [1, 2, 3],               // numbers, not strings
    });
    if (result.type !== "citation") throw new Error("expected citation");
    expect(result.chunk_ids).toEqual(["1", "2", "3"]);
  });

  test("non-citation events fall back to paragraph_break (defensive)", () => {
    const result = coerceCitationEvent({ type: "delta", content: "x" });
    expect(result).toEqual({ type: "paragraph_break" });
  });
});

describe("coerceContentBlock", () => {
  // Persisted content_blocks (metadata.content_blocks on history reload)
  // carry the integer sections.id on citation blocks, same as the live SSE
  // citation event. coerceContentBlock normalizes them on reload so the
  // citation jump works on reloaded conversations too.
  test("coerces a citation block's numeric section_id to string", () => {
    const result = coerceContentBlock({
      type: "citation", index: 2, section_id: 9, source_id: "src-1",
      timestamp_start: 12, chunk_ids: ["c1"],
    });
    if (result.type !== "citation") throw new Error("expected citation");
    expect(result.section_id).toBe("9");
  });

  test("passes a text block through unchanged", () => {
    expect(coerceContentBlock({ type: "text", text: "hello" })).toEqual({
      type: "text", text: "hello",
    });
  });

  test("passes a paragraph_break through", () => {
    expect(coerceContentBlock({ type: "paragraph_break" })).toEqual({ type: "paragraph_break" });
  });
});

describe("buildMindmapAskMessage", () => {
  // Stub t: echo the chosen i18n key + its params so we can assert which
  // template the branch picked and that evidence is threaded. i18n
  // interpolation itself is LanguageContext's job, tested elsewhere.
  const t = (key: string, params?: Record<string, string | number>) =>
    `${key}|${JSON.stringify(params ?? {})}`;

  test("root node, no evidence → today's discuss message", () => {
    expect(buildMindmapAskMessage(t, "Topic", null, "")).toBe(
      `lab.mindMap.discuss|${JSON.stringify({ topic: "Topic" })}`,
    );
  });

  test("child node, no evidence → discussInContext (unchanged from today)", () => {
    expect(buildMindmapAskMessage(t, "Child", "Parent", "")).toBe(
      `lab.mindMap.discussInContext|${JSON.stringify({ topic: "Child", context: "Parent" })}`,
    );
  });

  test("evidence present on a child → discussRef wraps the base with the verbatim quote", () => {
    const base = `lab.mindMap.discussInContext|${JSON.stringify({ topic: "Child", context: "Parent" })}`;
    expect(buildMindmapAskMessage(t, "Child", "Parent", "verbatim quote")).toBe(
      `lab.mindMap.discussRef|${JSON.stringify({ base, evidence: "verbatim quote" })}`,
    );
  });

  test("evidence present on the root → discussRef wraps the discuss base", () => {
    const base = `lab.mindMap.discuss|${JSON.stringify({ topic: "Topic" })}`;
    expect(buildMindmapAskMessage(t, "Topic", null, "q")).toBe(
      `lab.mindMap.discussRef|${JSON.stringify({ base, evidence: "q" })}`,
    );
  });
});
