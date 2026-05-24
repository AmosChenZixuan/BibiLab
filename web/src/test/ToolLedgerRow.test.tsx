import { afterEach, describe, expect, test } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ToolLedgerRow } from "@/components/lists/ToolLedgerRow";
import { TOOL_DISPLAY } from "@/lib/tool-display";
import type { RetrievalCall, MetadataCall } from "@/lib/chat-utils";

afterEach(() => {
  cleanup();
});

function renderRow(props: React.ComponentProps<typeof ToolLedgerRow>) {
  return render(
    <LanguageProvider>
      <JobActivityProvider>
        <ToolLedgerRow {...props} />
      </JobActivityProvider>
    </LanguageProvider>,
  );
}

const BASE_CALL: RetrievalCall = {
  query: "长期情景记忆",
  mode: "survey",
  candidates_evaluated: 10,
  sources_with_hits: 1,
  sources_total: 16,
  source_coverage: [{ source_id: "s1", video_id: "v1", title: "Test Video" }],
  context: [
    {
      chunk_id: "c1",
      citation_index: 1,
      source_id: "s1",
      source_title: "Test Video",
      timestamp_start: 0,
      timestamp_end: 132,
      rerank_score: 4.53,
      preview: "面试官问在构建一个长期陪伴性AI角色时 如何设计…",
    },
    {
      chunk_id: "c2",
      citation_index: 2,
      source_id: "s1",
      source_title: "Test Video",
      timestamp_start: 132,
      timestamp_end: 300,
      rerank_score: 3.21,
      preview: "Another preview text",
    },
  ],
  dropped_by_gate: 0,
  reranked: true,
  scoped_pool_size: 10,
  gate_margin: 0.25,
};

// ---------- search row ----------
describe("search row", () => {
  test("collapsed shows source count and cited chunks", () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: BASE_CALL });
    expect(container.innerHTML).toContain("1 sources");
    expect(container.innerHTML).toContain("2 chunks cited");
  });

  test("expands on click showing cited chunks + metadata + chunk list", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: BASE_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    expect(toggle).not.toBeNull();
    await userEvent.click(toggle!);

    expect(container.innerHTML).toContain("2 chunks cited");
    expect(container.innerHTML).toContain("长期情景记忆");
    expect(container.innerHTML).toContain("Survey");
    expect(container.innerHTML).toContain("all 16 sources");
    expect(container.innerHTML).toContain("Test Video");
    expect(container.innerHTML).toContain("[1]");
  });

  test("streaming disables expand: no toggle button, collapsed only", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: BASE_CALL, streaming: true });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
    expect(container.innerHTML).toContain("长期情景记忆");
    expect(container.innerHTML).not.toContain("Another preview text");
  });

  test("toggle collapses", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: BASE_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]')!;
    await userEvent.click(toggle);
    await userEvent.click(toggle);
    expect(container.innerHTML).not.toContain("all 16 sources");
  });
});

// ---------- empty variant ----------
describe("empty search row", () => {
  const EMPTY_CALL: RetrievalCall = {
    ...BASE_CALL,
    context: [],
    dropped_by_gate: 3,
  };

  test("renders amber empty row with dropped count", () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: EMPTY_CALL });
    expect(container.innerHTML).toContain("0 chunks");
    expect(container.innerHTML).toContain("3 dropped");
  });

  test("expands to show metadata + Result line, no chunk list", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: EMPTY_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);

    expect(container.innerHTML).toContain("长期情景记忆");
    expect(container.innerHTML).toContain("Result");
    expect(container.innerHTML).not.toContain("[1]");
  });

  test("streaming empty variant is non-expandable", async () => {
    const emptyCall: RetrievalCall = { ...BASE_CALL, context: [], dropped_by_gate: 4 };
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: emptyCall, streaming: true });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
  });
});

// ---------- pending ----------
describe("pending rows", () => {
  test("pending retrieve row shows spinner + label", () => {
    const { container } = renderRow({
      config: TOOL_DISPLAY.retrieve,
      pending: { id: "p1", query: "pending query", mode: "narrow", tool_name: "retrieve" },
    });
    expect(container.innerHTML).toContain("retrieving…");
  });

  test("pending metadata row shows spinner + unified label", () => {
    const { container } = renderRow({
      config: TOOL_DISPLAY.query_list_metadata,
      pending: { id: "pm1", query_type: "count" },
    });
    expect(container.innerHTML).toContain("querying list…");
  });

  test("has no toggle button", () => {
    const { container } = renderRow({
      config: TOOL_DISPLAY.retrieve,
      pending: { id: "p1", query: "test", mode: "narrow", tool_name: "retrieve" },
    });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
  });
});

// ---------- metadata row ----------
describe("metadata row", () => {
  const META_CALL: MetadataCall = {
    name: "query_list_metadata",
    query_type: "count_sources",
    result: { source_count: 42 },
  };

  test("collapsed shows icon + label", () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.query_list_metadata, call: META_CALL });
    expect(container.innerHTML).toContain("Querying list info");
  });

  test("expands to show raw JSON", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.query_list_metadata, call: META_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);
    expect(container.innerHTML).toContain("source_count");
    expect(container.innerHTML).toContain("42");
  });

  test("streaming metadata row shows label, no expand", () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.query_list_metadata, call: META_CALL, streaming: true });
    expect(container.innerHTML).toContain("Querying list info");
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
  });
});

// ---------- mode rendering ----------
describe("mode rendering", () => {
  test('null mode renders "—"', async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: { ...BASE_CALL, mode: null } });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);
    expect(container.innerHTML).toContain("—");
  });

  test('"narrow" renders "Narrow"', async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: { ...BASE_CALL, mode: "narrow" } });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);
    expect(container.innerHTML).toContain("Narrow");
  });
});

const NO_MATCH = {
  sequence_number: 8,
  season_number: null,
  matched_count: 0,
  no_match: true,
};

describe("facet no-match hint (#319)", () => {
  test("default collapsed shows amber warning icon with hint aria-label", () => {
    renderRow({ config: TOOL_DISPLAY.retrieve, call: { ...BASE_CALL, facet_scope: NO_MATCH } });
    const icon = screen.getByLabelText(
      "No source matched #8 — searched all sources instead.",
    );
    expect(icon).toBeTruthy();
    expect(screen.getByText(/1 sources/)).toBeTruthy();
  });

  test("default expanded shows amber Facet detail line", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY.retrieve, call: { ...BASE_CALL, facet_scope: NO_MATCH } });
    await userEvent.click(container.querySelector('button[aria-label="Toggle retrieval details"]')!);
    expect(container.innerHTML).toContain("Facet");
    expect(container.innerHTML).toContain("No source matched #8 — searched all sources instead.");
  });

  test("no hint when no_match is false", () => {
    renderRow({
      config: TOOL_DISPLAY.retrieve,
      call: { ...BASE_CALL, facet_scope: { ...NO_MATCH, no_match: false } },
    });
    expect(screen.queryByLabelText(/No source matched/)).toBeNull();
  });

  test("no hint when facet_scope absent (legacy message)", () => {
    renderRow({ config: TOOL_DISPLAY.retrieve, call: BASE_CALL });
    expect(screen.queryByLabelText(/No source matched/)).toBeNull();
  });

  test("streaming default still shows the icon (visible without expand)", () => {
    renderRow({ config: TOOL_DISPLAY.retrieve, call: { ...BASE_CALL, facet_scope: NO_MATCH }, streaming: true });
    expect(
      screen.getByLabelText("No source matched #8 — searched all sources instead."),
    ).toBeTruthy();
  });

  test("empty variant also surfaces the hint", () => {
    const emptyCall = { ...BASE_CALL, context: [], dropped_by_gate: 3, facet_scope: NO_MATCH };
    renderRow({ config: TOOL_DISPLAY.retrieve, call: emptyCall });
    expect(
      screen.getByLabelText("No source matched #8 — searched all sources instead."),
    ).toBeTruthy();
  });
});
