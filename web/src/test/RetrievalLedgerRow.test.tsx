import { afterEach, describe, expect, test } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { RetrievalLedgerRow } from "@/components/lists/RetrievalLedgerRow";
import type { RetrievalCall } from "@/lib/chat-utils";

afterEach(() => {
  cleanup();
});

function renderRow(props: React.ComponentProps<typeof RetrievalLedgerRow>) {
  return render(
    <LanguageProvider>
      <JobActivityProvider>
        <RetrievalLedgerRow {...props} />
      </JobActivityProvider>
    </LanguageProvider>,
  );
}

const BASE_CALL: RetrievalCall = {
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
  scope_choice: "exclude",
  excluded_count: 6,
  scoped_pool_size: 10,
  gate_margin: 0.25,
  reused_from_prior_call_id: null,
};

// ---------- variant: default ----------
describe("variant=default", () => {
  test("renders default row with summary text", () => {
    const { container } = renderRow({ variant: "default", call: BASE_CALL });
    expect(container.innerHTML).toContain("2 chunks");
    expect(container.innerHTML).toContain("1 source");
  });

  test("expands on click showing metadata + chunk list", async () => {
    const { container } = renderRow({ variant: "default", call: BASE_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    expect(toggle).not.toBeNull();
    await userEvent.click(toggle!);

    // After expand: metadata is visible
    expect(container.innerHTML).toContain("长期情景记忆");
    expect(container.innerHTML).toContain("many");
    expect(container.innerHTML).toContain("excluded 6 of 16");
    // chunk list — source_title is "Test Video" (exact case)
    expect(container.innerHTML).toContain("Test Video");
    expect(container.innerHTML).toContain("[1]");
  });

  test("streaming disables expand: no toggle button, collapsed only", async () => {
    const { container } = renderRow({ variant: "default", call: BASE_CALL, streaming: true });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
    // collapsed summary still shows the query; chunk previews stay hidden
    expect(container.innerHTML).toContain("长期情景记忆");
    expect(container.innerHTML).not.toContain("Another preview text");
  });

  test("streaming empty variant is non-expandable", async () => {
    const emptyCall: RetrievalCall = { ...BASE_CALL, context: [], dropped_by_gate: 4 };
    const { container } = renderRow({ variant: "empty", call: emptyCall, streaming: true });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
  });

  test("toggle collapses", async () => {
    const { container } = renderRow({ variant: "default", call: BASE_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]')!;
    await userEvent.click(toggle);
    await userEvent.click(toggle);
    // After re-collapse: metadata should not be visible
    expect(container.innerHTML).not.toContain("excluded 6 of 16");
  });
});

// ---------- variant: empty ----------
describe("variant=empty", () => {
  const EMPTY_CALL: RetrievalCall = {
    ...BASE_CALL,
    context: [],
    dropped_by_gate: 3,
  };

  test("renders amber empty row with dropped count", () => {
    const { container } = renderRow({ variant: "empty", call: EMPTY_CALL });
    expect(container.innerHTML).toContain("0 chunks");
    expect(container.innerHTML).toContain("3 dropped");
  });

  test("expands to show metadata + Result line, no chunk list", async () => {
    const { container } = renderRow({ variant: "empty", call: EMPTY_CALL });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);

    expect(container.innerHTML).toContain("长期情景记忆");
    expect(container.innerHTML).toContain("Result");
    expect(container.innerHTML).not.toContain("[1]");
  });
});

// ---------- variant: reused ----------
describe("variant=reused", () => {
  const REUSED_CALL: RetrievalCall = {
    ...BASE_CALL,
    query: "(reused)",
    context: [],
    reused_from_prior_call_id: "prior-call-id",
  };

  test("renders single-line reused row", () => {
    const { container } = renderRow({ variant: "reused", call: REUSED_CALL });
    expect(container.innerHTML).toContain("reused from previous turn");
  });

  test("has no toggle button", () => {
    const { container } = renderRow({ variant: "reused", call: REUSED_CALL });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
  });
});

// ---------- variant: pending ----------
describe("variant=pending", () => {
  test("pending retrieve row shows spinner + label", () => {
    const { container } = renderRow({
      variant: "pending",
      pending: { id: "p1", query: "pending query", expected_hits: "few" },
    });
    expect(container.innerHTML).toContain("retrieving…");
  });

  test("pending metadata row shows spinner + query_type label", () => {
    const { container } = renderRow({
      variant: "pending",
      pending: { id: "pm1", query_type: "count" },
    });
    expect(container.innerHTML).toContain("counting sources");
  });

  test("has no toggle button", () => {
    const { container } = renderRow({
      variant: "pending",
      pending: { id: "p1", query: "test", expected_hits: "one" },
    });
    expect(container.querySelector('button[aria-label="Toggle retrieval details"]')).toBeNull();
  });
});

// ---------- scope_choice mapping ----------
describe("scope_choice rendering", () => {
  test('scope_choice=exclude renders "excluded N of T"', async () => {
    const { container } = renderRow({ variant: "default", call: { ...BASE_CALL, scope_choice: "exclude", excluded_count: 6, scoped_pool_size: 10 } });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);
    expect(container.innerHTML).toContain("excluded 6 of 16");
  });

  test('scope_choice=whitelist renders "only N of T"', async () => {
    const { container } = renderRow({ variant: "default", call: { ...BASE_CALL, scope_choice: "whitelist", excluded_count: null, scoped_pool_size: 3 } });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);
    expect(container.innerHTML).toContain("only 3 of 16");
  });

  test('scope_choice=none renders "all N sources"', async () => {
    const { container } = renderRow({ variant: "default", call: { ...BASE_CALL, scope_choice: "none", excluded_count: null, scoped_pool_size: 16 } });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);
    expect(container.innerHTML).toContain("all 16 sources");
  });
});

// ---------- mode rendering ----------
describe("mode rendering", () => {
  test('null expected_hits renders "—"', async () => {
    const { container } = renderRow({ variant: "default", call: { ...BASE_CALL, expected_hits: null } });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);
    expect(container.innerHTML).toContain("—");
  });

  test('"one" renders "one"', async () => {
    const { container } = renderRow({ variant: "default", call: { ...BASE_CALL, expected_hits: "one" } });
    const toggle = container.querySelector('button[aria-label="Toggle retrieval details"]');
    await userEvent.click(toggle!);
    expect(container.innerHTML).toContain("one");
  });
});
