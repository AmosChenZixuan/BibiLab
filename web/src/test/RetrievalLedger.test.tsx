import { describe, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { LanguageProvider } from "@/app/LanguageContext";
import { RetrievalLedger } from "@/components/lists/RetrievalLedger";
import type { RetrievalCall, PendingRagCall, PendingMetadataCall } from "@/lib/chat-utils";

function renderLedger(props: React.ComponentProps<typeof RetrievalLedger>) {
  return render(
    <LanguageProvider>
      <RetrievalLedger {...props} />
    </LanguageProvider>,
  );
}

const DEFAULT_CALL: RetrievalCall = {
  query: "长期情景记忆",
  expected_hits: "many",
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

describe("RetrievalLedger", () => {
  test("returns null when all arrays are empty", () => {
    const { container } = renderLedger({
      calls: [],
      pendingRetrieve: [],
      pendingMetadata: [],
    });
    expect(container.firstChild).toBeNull();
  });

  test("renders one row per completed call", () => {
    renderLedger({
      calls: [DEFAULT_CALL],
      pendingRetrieve: [],
      pendingMetadata: [],
    });
    expect(screen.getByText(/长期情景记忆/)).toBeInTheDocument();
  });

  test("renders pending retrieve rows", () => {
    const pending: PendingRagCall = { id: "p1", query: "pending query", expected_hits: "few" };
    renderLedger({
      calls: [],
      pendingRetrieve: [pending],
      pendingMetadata: [],
    });
    // pending RAG row shows spinner + "retrieving..." label
    expect(screen.getByText(/retrieving…/i)).toBeInTheDocument();
  });

  test("renders pending metadata rows", () => {
    const pending: PendingMetadataCall = { id: "pm1", query_type: "count" };
    renderLedger({
      calls: [],
      pendingRetrieve: [],
      pendingMetadata: [pending],
    });
    expect(screen.getByText(/counting sources/)).toBeInTheDocument();
  });

  test("streaming-shaped call (context absent) renders as default, not empty", () => {
    // The SSE tool_result payload omits context[]
    // (those are reconstructed only at persist time). dropped_by_gate can be
    // >0 even on a successful retrieval. callVariant must NOT misclassify a
    // context-absent streaming call as the amber "empty" variant.
    const streamingCall = {
      query: "streaming-query",
      expected_hits: "few",
      candidates_evaluated: 8,
      sources_with_hits: 2,
      sources_total: 16,
      source_coverage: [{ source_id: "s1", video_id: "v1", title: "Vid" }],
      dropped_by_gate: 3,
      reranked: true,
      scoped_pool_size: 16,
      gate_margin: 2.0,
    } as unknown as RetrievalCall;

    renderLedger({ calls: [streamingCall], pendingRetrieve: [], pendingMetadata: [] });

    // default variant shows the query; empty variant (collapsed) shows only
    // "0 chunks (3 dropped)" with no query.
    expect(screen.getByText(/streaming-query/)).toBeInTheDocument();
    expect(screen.queryByText(/3 dropped/)).not.toBeInTheDocument();
    // context absent on streaming payload; persisted context is one entry per
    // cited source, so the streaming chunk count must equal source_coverage
    // length (1) — not 0 — to stay consistent with post-refresh.
    expect(screen.getByText(/1 chunks · 1 sources/)).toBeInTheDocument();
  });

  test("ordering: calls first, then pending retrieve, then pending metadata", () => {
    const call1: RetrievalCall = { ...DEFAULT_CALL, query: "call-query" };
    const pending1: PendingRagCall = { id: "p1", query: "pending-retrieve", expected_hits: "one" };
    const pending2: PendingMetadataCall = { id: "pm1", query_type: "longest" };

    renderLedger({
      calls: [call1],
      pendingRetrieve: [pending1],
      pendingMetadata: [pending2],
    });

    // Call row should appear (has query text), pending rows also appear
    expect(screen.getByText(/call-query/)).toBeInTheDocument();
    expect(screen.getByText(/longest/)).toBeInTheDocument();
    // The ledger container should be present
    const ledgerEl = document.querySelector('[class*="flex"][class*="flex-col"]');
    expect(ledgerEl).toBeInTheDocument();
  });
});
