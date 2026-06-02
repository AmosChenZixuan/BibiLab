import { afterEach, describe, expect, test } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ToolLedgerRow } from "@/components/lists/ToolLedgerRow";
import { FIND_PASSAGES_TOOL_NAME, READ_SOURCE_TOOL_NAME, TOOL_DISPLAY } from "@/lib/tool-display";
import type { RetrievalCall, PendingRagCall } from "@/lib/chat-utils";

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
  tool_name: FIND_PASSAGES_TOOL_NAME,
  candidates_evaluated: 10,
  sources_with_hits: 1,
  sources_total: 16,
  source_coverage: [{ source_id: "s1", title: "Test Video" }],
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
  reranked: true,
  scoped_pool_size: 10,
};

// ---------- search row ----------
describe("search row", () => {
  test("collapsed shows source count and cited chunks", () => {
    const { container } = renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: BASE_CALL });
    expect(container.innerHTML).toContain("1 sources");
    expect(container.innerHTML).toContain("2 chunks cited");
  });

  test("expands on click showing cited chunks + metadata + chunk list", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: BASE_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    expect(toggle).not.toBeNull();
    await userEvent.click(toggle!);

    expect(container.innerHTML).toContain("2 chunks cited");
    expect(container.innerHTML).toContain("长期情景记忆");
    expect(container.innerHTML).toContain("all 16 sources");
    expect(container.innerHTML).toContain("Test Video");
    expect(container.innerHTML).toContain("[1]");
  });

  test("streaming disables expand: no toggle button, collapsed only", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: BASE_CALL, streaming: true });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
    expect(container.innerHTML).toContain("长期情景记忆");
    expect(container.innerHTML).not.toContain("Another preview text");
  });

  test("toggle collapses", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: BASE_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]')!;
    await userEvent.click(toggle);
    await userEvent.click(toggle);
    expect(container.innerHTML).not.toContain("all 16 sources");
  });
});

// ---------- pending ----------
describe("pending rows", () => {
  test("pending find_passages row shows spinner + summaryPending label", () => {
    const { container } = renderRow({
      config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME],
      pending: { id: "p1", query: "pending query", tool_name: FIND_PASSAGES_TOOL_NAME },
    });
    expect(container.innerHTML).toContain("finding passages…");
  });

  test("pending row has no toggle button", () => {
    const { container } = renderRow({
      config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME],
      pending: { id: "p1", query: "test", tool_name: FIND_PASSAGES_TOOL_NAME },
    });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
  });
});

// ---------- read_source completed row ----------
describe("read_source completed row", () => {
  const READ_CALL: RetrievalCall = {
    query: "",
    tool_name: READ_SOURCE_TOOL_NAME,
    candidates_evaluated: 0,
    sources_with_hits: 0,
    sources_total: 1,
    source_coverage: [],
    context: [],
    reranked: false,
    scoped_pool_size: 1,
    source_id: "s1",
    source_title: "Ep 4",
  };

  test("renders a read_source label chip with source_title, no per-chunk list, non-expandable", () => {
    const { container } = renderRow({ config: TOOL_DISPLAY[READ_SOURCE_TOOL_NAME], call: READ_CALL });
    // Read in full label is visible
    expect(screen.getByText(/read in full/i)).toBeInTheDocument();
    // Source title is visible
    expect(screen.getByText("Ep 4")).toBeInTheDocument();
    // No expand button (read_source has no toggle)
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
    // No chunk list rendered: no timestamp ranges, no rerank scores
    expect(container.innerHTML).not.toContain("4.53");
    expect(container.innerHTML).not.toContain("0:00");
  });
});

// ---------- read_source pending row ----------
describe("read_source pending row", () => {
  test("renders a read_source pending chip with the label but no query text", () => {
    const pending: PendingRagCall = { id: "p1", tool_name: READ_SOURCE_TOOL_NAME, query: "" };
    const { container } = renderRow({ config: TOOL_DISPLAY[READ_SOURCE_TOOL_NAME], pending });
    // Label visible
    expect(screen.getByText(/read in full/i)).toBeInTheDocument();
    // No query text rendered
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
    // Spinner present
    expect(container.querySelector(".animate-spin")).not.toBeNull();
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
    renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: { ...BASE_CALL, facet_scope: NO_MATCH } });
    const icon = screen.getByLabelText(
      "No source matched #8 — searched all sources instead.",
    );
    expect(icon).toBeTruthy();
    expect(screen.getByText(/1 sources/)).toBeTruthy();
  });

  test("default expanded shows amber Facet detail line", async () => {
    const { container } = renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: { ...BASE_CALL, facet_scope: NO_MATCH } });
    await userEvent.click(container.querySelector('button[aria-label="Toggle retrieval details"]')!);
    expect(container.innerHTML).toContain("Facet");
    expect(container.innerHTML).toContain("No source matched #8 — searched all sources instead.");
  });

  test("no hint when no_match is false", () => {
    renderRow({
      config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME],
      call: { ...BASE_CALL, facet_scope: { ...NO_MATCH, no_match: false } },
    });
    expect(screen.queryByLabelText(/No source matched/)).toBeNull();
  });

  test("no hint when facet_scope absent (legacy message)", () => {
    renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: BASE_CALL });
    expect(screen.queryByLabelText(/No source matched/)).toBeNull();
  });

  test("streaming default still shows the icon (visible without expand)", () => {
    renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: { ...BASE_CALL, facet_scope: NO_MATCH }, streaming: true });
    expect(
      screen.getByLabelText("No source matched #8 — searched all sources instead."),
    ).toBeTruthy();
  });

  test("empty variant also surfaces the hint", () => {
    const emptyCall = { ...BASE_CALL, context: [], facet_scope: NO_MATCH };
    renderRow({ config: TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME], call: emptyCall });
    expect(
      screen.getByLabelText("No source matched #8 — searched all sources instead."),
    ).toBeTruthy();
  });
});
