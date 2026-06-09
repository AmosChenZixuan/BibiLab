import { describe, expect, test } from "vitest";

import type { ContentBlock } from "@/lib/chat-utils";
import {
  autoResize,
  facetNoMatchHint,
  formatDurationHuman,
  formatSubtitle,
  formatTimestamp,
  stripLegacyTokens,
  RetrievalCall,
} from "@/lib/chat-utils";
import { READ_SECTION_TOOL_NAME } from "@/lib/tool-display";

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
      tool_name: "find_passages",
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
    expect(call.tool_name).toBe("find_passages");
    expect(call.context![0].chunk_id).toBe("c1");
    expect(call.context![0].citation_index).toBe(1);
    expect(call.context![0].timestamp_start).toBe(120.4);
    expect(call.context![0].rerank_score).toBe(0.95);
  });

  test("context can be empty array", () => {
    const call: RetrievalCall = {
      query: "test",
      tool_name: "find_passages",
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

describe("section-grained types", () => {
  test("ContentBlock citation carries section_id + timestamp_start", () => {
    const b: ContentBlock = {
      type: "citation", index: 1, section_id: "sec-1", source_id: "src-1",
      timestamp_start: 42, chunk_ids: ["c1"],
    };
    expect(b.section_id).toBe("sec-1");
    expect(b.timestamp_start).toBe(42);
  });

  test("read_section tool name constant", () => {
    expect(READ_SECTION_TOOL_NAME).toBe("read_section");
  });

  test("RetrievalCall uses section_coverage", () => {
    const call: RetrievalCall = {
      query: "q", tool_name: "find_passages", candidates_evaluated: 0,
      sources_with_hits: 0, sources_total: 1, reranked: false, scoped_pool_size: 1,
      section_coverage: [{
        section_id: "s1", source_id: "src", source_title: "T",
        seq: 1, timestamp_start: 0, timestamp_end: 60,
      }],
    };
    expect(call.section_coverage[0].section_id).toBe("s1");
  });
});
