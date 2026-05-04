import { describe, expect, test } from "vitest";

import {
  autoResize,
  formatDurationHuman,
  formatSubtitle,
  formatTimestamp,
  parseCitations,
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

describe("parseCitations", () => {
  test("extracts citations and cleans content", () => {
    const text = "Some text [Video Title @ 10s-20s] more text";
    const { citations, cleanContent } = parseCitations(text);
    expect(citations).toEqual([
      { source_title: "Video Title", timestamp_start: 10, timestamp_end: 20 },
    ]);
    expect(cleanContent).toBe("Some text  more text");
  });

  test("multiple citations", () => {
    const text = "[A @ 0s-5s] and [B @ 60s-120s]";
    const { citations, cleanContent } = parseCitations(text);
    expect(citations).toHaveLength(2);
    expect(citations[0].source_title).toBe("A");
    expect(citations[1].source_title).toBe("B");
    expect(cleanContent).toBe(" and ");
  });

  test("deduplicates repeated citations", () => {
    const text = "[A @ 0s-5s] and [A @ 0s-5s] again [B @ 10s-20s]";
    const { citations, cleanContent } = parseCitations(text);
    expect(citations).toHaveLength(2);
    expect(citations[0]).toEqual({ source_title: "A", timestamp_start: 0, timestamp_end: 5 });
    expect(citations[1]).toEqual({ source_title: "B", timestamp_start: 10, timestamp_end: 20 });
    expect(cleanContent).toBe(" and  again ");
  });

  test("does not deduplicate different timestamps for same title", () => {
    const text = "[A @ 0s-5s] and [A @ 10s-20s]";
    const { citations } = parseCitations(text);
    expect(citations).toHaveLength(2);
  });

  test("no citations", () => {
    const { citations, cleanContent } = parseCitations("plain text");
    expect(citations).toEqual([]);
    expect(cleanContent).toBe("plain text");
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
