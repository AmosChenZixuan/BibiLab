import { afterEach, describe, expect, test } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { LanguageProvider } from "@/app/LanguageContext";
import { ToolLedger } from "@/components/lists/ToolLedger";
import type { RetrievalCall, MetadataCall, PendingRagCall, PendingMetadataCall } from "@/lib/chat-utils";

afterEach(() => {
  cleanup();
});

function renderLedger(props: React.ComponentProps<typeof ToolLedger>) {
  return render(
    <LanguageProvider>
      <ToolLedger {...props} />
    </LanguageProvider>,
  );
}

const DEFAULT_CALL: RetrievalCall = {
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

describe("ToolLedger", () => {
  test("returns null when all arrays are empty", () => {
    const { container } = renderLedger({
      ragCalls: [],
      metadataCalls: [],
      pendingRagCalls: [],
      pendingMetadataCalls: [],
    });
    expect(container.firstChild).toBeNull();
  });

  test("renders one row per completed search call", () => {
    renderLedger({
      ragCalls: [DEFAULT_CALL],
      metadataCalls: [],
      pendingRagCalls: [],
      pendingMetadataCalls: [],
    });
    expect(screen.getByText(/长期情景记忆/)).toBeInTheDocument();
  });

  test("renders pending retrieve rows", () => {
    const pending: PendingRagCall = { id: "p1", query: "pending query", mode: "narrow", tool_name: "retrieve" };
    renderLedger({
      ragCalls: [],
      metadataCalls: [],
      pendingRagCalls: [pending],
      pendingMetadataCalls: [],
    });
    expect(screen.getByText(/retrieving…/i)).toBeInTheDocument();
  });

  test("renders pending metadata rows with unified label", () => {
    const pending: PendingMetadataCall = { id: "pm1", query_type: "count" };
    renderLedger({
      ragCalls: [],
      metadataCalls: [],
      pendingRagCalls: [],
      pendingMetadataCalls: [pending],
    });
    expect(screen.getByText(/querying list…/i)).toBeInTheDocument();
  });

  test("renders completed metadata rows", () => {
    const metaCall: MetadataCall = {
      name: "query_list_metadata",
      query_type: "count_sources",
      result: { source_count: 42 },
    };
    renderLedger({
      ragCalls: [],
      metadataCalls: [metaCall],
      pendingRagCalls: [],
      pendingMetadataCalls: [],
    });
    expect(screen.getByText(/Querying list info/i)).toBeInTheDocument();
  });

  test("metadata row expand shows raw JSON", async () => {
    const user = userEvent.setup();
    const metaCall: MetadataCall = {
      name: "query_list_metadata",
      query_type: "count_sources",
      result: { source_count: 42 },
    };
    renderLedger({
      ragCalls: [],
      metadataCalls: [metaCall],
      pendingRagCalls: [],
      pendingMetadataCalls: [],
    });
    const toggle = screen.getByRole("button");
    await user.click(toggle);
    expect(screen.getByText(/"source_count"/)).toBeInTheDocument();
  });

  test("search row collapsed shows source count only, no chunk count", () => {
    renderLedger({
      ragCalls: [DEFAULT_CALL],
      metadataCalls: [],
      pendingRagCalls: [],
      pendingMetadataCalls: [],
    });
    expect(screen.getByText(/1 sources/)).toBeInTheDocument();
    expect(screen.queryByText(/2 chunks/)).not.toBeInTheDocument();
  });

  test("search row expanded shows cited chunks", async () => {
    const user = userEvent.setup();
    renderLedger({
      ragCalls: [DEFAULT_CALL],
      metadataCalls: [],
      pendingRagCalls: [],
      pendingMetadataCalls: [],
    });
    const toggle = screen.getByRole("button");
    await user.click(toggle);
    expect(screen.getByText(/2 chunks cited/)).toBeInTheDocument();
  });

  test("streaming-shaped call (context absent) renders as default, not empty", () => {
    const streamingCall = {
      query: "streaming-query",
      mode: "narrow",
      candidates_evaluated: 8,
      sources_with_hits: 2,
      sources_total: 16,
      source_coverage: [{ source_id: "s1", video_id: "v1", title: "Vid" }],
      dropped_by_gate: 3,
      reranked: true,
      scoped_pool_size: 16,
      gate_margin: 2.0,
    } as unknown as RetrievalCall;

    renderLedger({ ragCalls: [streamingCall], metadataCalls: [], pendingRagCalls: [], pendingMetadataCalls: [] });

    expect(screen.getByText(/streaming-query/)).toBeInTheDocument();
    expect(screen.queryByText(/3 dropped/)).not.toBeInTheDocument();
    expect(screen.getByText(/1 sources/)).toBeInTheDocument();
  });

  test("ordering: search calls, then metadata calls, then pending", () => {
    const ragCall: RetrievalCall = { ...DEFAULT_CALL, query: "rag-query" };
    const metaCall: MetadataCall = { name: "query_list_metadata", query_type: "count_sources", result: {} };
    const pending: PendingRagCall = { id: "p1", query: "pending-retrieve", mode: "narrow", tool_name: "retrieve" };

    renderLedger({
      ragCalls: [ragCall],
      metadataCalls: [metaCall],
      pendingRagCalls: [pending],
      pendingMetadataCalls: [],
    });

    expect(screen.getByText(/rag-query/)).toBeInTheDocument();
    expect(screen.getByText(/Querying list info/)).toBeInTheDocument();
    expect(screen.getByText(/retrieving…/)).toBeInTheDocument();
  });
});
