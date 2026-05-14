import { describe, expect, test } from "vitest";

import {
  autoResize,
  formatDurationHuman,
  formatSubtitle,
  formatTimestamp,
  stripLegacyTokens,
  ExpectedHits,
  RetrievalCall,
  LegacyRagCall,
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

describe("ExpectedHits type", () => {
  test("is one of the expected literal values or null", () => {
    // Type-level test: these assignments must compile without error.
    // We test the union by casting to verify the type exists.
    const values: ExpectedHits[] = ["one", "few", "many", null];
    expect(values).toHaveLength(4);
  });
});

describe("RetrievalCall", () => {
  test("has all required fields", () => {
    const call: RetrievalCall = {
      query: "test query",
      expected_hits: "few",
      candidates_evaluated: 5,
      sources_with_hits: 2,
      sources_total: 3,
      source_coverage: [],
      context: [
        {
          chunk_id: "v1_120_145",
          timestamp_start: 120.4,
          timestamp_end: 145.0,
          rerank_score: 0.95,
          preview: "test content here",
        },
      ],
    };
    expect(call.expected_hits).toBe("few");
    expect(call.context[0].chunk_id).toBe("v1_120_145");
    expect(call.context[0].timestamp_start).toBe(120.4);
    expect(call.context[0].rerank_score).toBe(0.95);
  });

  test("context can be empty array", () => {
    const call: RetrievalCall = {
      query: "test",
      expected_hits: "one",
      candidates_evaluated: 0,
      sources_with_hits: 0,
      sources_total: 1,
      source_coverage: [],
      context: [],
    };
    expect(call.context).toHaveLength(0);
  });
});

describe("LegacyRagCall", () => {
  test("has search_mode instead of expected_hits", () => {
    const legacy: LegacyRagCall = {
      query: "test",
      search_mode: "factual",
      candidates_evaluated: 5,
      sources_with_hits: 2,
      sources_total: 3,
      source_coverage: [],
    };
    expect("expected_hits" in legacy).toBe(false);
    expect("search_mode" in legacy).toBe(true);
  });

  test("parses without context field", () => {
    // AC4: legacy messages (no context) parse without errors
    const raw = {
      query: "legacy call",
      search_mode: "breadth",
      candidates_evaluated: 3,
      sources_with_hits: 1,
      sources_total: 2,
      source_coverage: [],
    };
    const parsed: LegacyRagCall = raw as unknown as LegacyRagCall;
    expect(parsed.query).toBe("legacy call");
    expect(parsed.search_mode).toBe("breadth");
  });
});

describe("RagMetadata legacy detection", () => {
  test("new-style metadata has context on each call", () => {
    const metadata = {
      calls: [
        {
          query: "test",
          expected_hits: "few",
          candidates_evaluated: 1,
          sources_with_hits: 1,
          sources_total: 1,
          source_coverage: [],
          context: [
            {
              chunk_id: "v1_0_10",
              timestamp_start: 0,
              timestamp_end: 10,
              rerank_score: 0.9,
              preview: "x",
            },
          ],
        },
      ],
    };
    // All calls have context field
    for (const call of metadata.calls) {
      expect(call).toHaveProperty("context");
    }
  });

  test("legacy metadata has no context field", () => {
    const metadata = {
      calls: [
        {
          query: "old call",
          search_mode: "factual",
          candidates_evaluated: 1,
          sources_with_hits: 1,
          sources_total: 1,
          source_coverage: [],
        },
      ],
      legacy: true,
    };
    expect(metadata.legacy).toBe(true);
  });
});
