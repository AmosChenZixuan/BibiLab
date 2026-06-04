import { afterEach, describe, expect, test } from "vitest";
import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { LanguageProvider } from "@/app/LanguageContext";
import { ToolLedger } from "@/components/lists/ToolLedger";
import type { RetrievalCall, PendingRagCall } from "@/lib/chat-utils";
import { FIND_PASSAGES_TOOL_NAME } from "@/lib/tool-display";
import { renderWithProviders } from "@/test/utils";

afterEach(() => {
  cleanup();
});

function renderLedger(props: React.ComponentProps<typeof ToolLedger>) {
  return renderWithProviders(<ToolLedger {...props} />, {
    providers: [LanguageProvider],
  });
}

const DEFAULT_CALL: RetrievalCall = {
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

describe("ToolLedger", () => {
  test("returns null when all arrays are empty", () => {
    const { container } = renderLedger({
      ragCalls: [],
      pendingRagCalls: [],
    });
    expect(container.firstChild).toBeNull();
  });

  test("renders one row per completed search call", () => {
    renderLedger({
      ragCalls: [DEFAULT_CALL],
      pendingRagCalls: [],
    });
    expect(screen.getByText(/长期情景记忆/)).toBeInTheDocument();
  });

  test("renders pending find_passages rows with summaryPending label", () => {
    const pending: PendingRagCall = { id: "p1", query: "pending query", tool_name: FIND_PASSAGES_TOOL_NAME };
    renderLedger({
      ragCalls: [],
      pendingRagCalls: [pending],
    });
    expect(screen.getByText(/finding passages…/i)).toBeInTheDocument();
  });

  test("search row collapsed shows source count and cited chunks", () => {
    renderLedger({
      ragCalls: [DEFAULT_CALL],
      pendingRagCalls: [],
    });
    expect(screen.getByText(/1 sources/)).toBeInTheDocument();
    expect(screen.getByText(/2 chunks cited/)).toBeInTheDocument();
  });

  test("streaming-shaped call (context absent) renders without crash", () => {
    const streamingCall = {
      query: "streaming-query",
      tool_name: FIND_PASSAGES_TOOL_NAME,
      candidates_evaluated: 8,
      sources_with_hits: 2,
      sources_total: 16,
      source_coverage: [{ source_id: "s1", title: "Vid" }],
      reranked: true,
      scoped_pool_size: 16,
    } as unknown as RetrievalCall;

    renderLedger({ ragCalls: [streamingCall], pendingRagCalls: [] });

    expect(screen.getByText(/streaming-query/)).toBeInTheDocument();
    expect(screen.getByText(/1 sources/)).toBeInTheDocument();
  });

  test("ordering: completed search calls render before pending", () => {
    const ragCall: RetrievalCall = { ...DEFAULT_CALL, query: "rag-query" };
    const pending: PendingRagCall = { id: "p1", query: "pending-retrieve", tool_name: FIND_PASSAGES_TOOL_NAME };

    const { container } = renderLedger({
      ragCalls: [ragCall],
      pendingRagCalls: [pending],
    });

    expect(screen.getByText(/rag-query/)).toBeInTheDocument();
    expect(screen.getByText(/finding passages…/)).toBeInTheDocument();
    // Completed call appears before the pending one in the rendered DOM
    const ragIndex = container.innerHTML.indexOf("rag-query");
    const pendingIndex = container.innerHTML.indexOf("finding passages…");
    expect(ragIndex).toBeLessThan(pendingIndex);
  });
});
