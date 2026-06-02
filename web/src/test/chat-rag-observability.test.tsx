import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ChatPanel } from "@/components/lists/ChatPanel";
import type { Source } from "@/lib/types";

const SOURCE_1: Source = {
  id: "src-1",
  video_id: "BV1test",
  platform: "bilibili",
  title: "Test Video A",
  summary: "A test video",
  keywords: [],
  cover_url: null,
  source_url: "https://bilibili.com/video/BV1test",
  duration_seconds: 3600,
  uploader: "TestUploader",
  language: "en",
  processed_at: "2026-04-08T12:00:00Z",
};

function makeSseStream(events: string[]) {
  const body = new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(new TextEncoder().encode(event));
      }
      controller.close();
    },
  });
  return new Response(body, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

function renderChatPanel(props?: Partial<React.ComponentProps<typeof ChatPanel>>) {
  return render(
    <LanguageProvider>
      <JobActivityProvider>
        <ChatPanel
          selectedSourceIds={[]}
          sources={[]}
          listId="list-1"
          {...props}
        />
      </JobActivityProvider>
    </LanguageProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("RAG observability via SSE tool_result", () => {
  test("tool_result event attaches rag to in-progress message", async () => {
    vi.spyOn(window, "fetch").mockImplementation(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"tool_result","name":"find_passages","result":{"query":"A","tool_name":"find_passages","candidates_evaluated":30,"sources_with_hits":1,"sources_total":1,"source_coverage":[{"source_id":"s1","title":"Test Video A"}],"context":[{"chunk_id":"c1","citation_index":1,"source_id":"s1","source_title":"Test Video A","timestamp_start":0,"timestamp_end":132,"rerank_score":4.53,"preview":"test preview"}],"reranked":true,"scoped_pool_size":1}}\n\n',
          'data: {"type":"delta","content":"Hello"}\n\n',
          'data: {"type":"done"}\n\n',
        ]),
      ),
    );

    renderChatPanel({
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hi");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Hello")).toBeInTheDocument();
    });
  });

  test("ToolLedger renders above bubble with one row per call", async () => {
    vi.spyOn(window, "fetch").mockImplementation(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"tool_result","name":"find_passages","result":{"query":"长期情景记忆","tool_name":"find_passages","candidates_evaluated":10,"sources_with_hits":1,"sources_total":16,"source_coverage":[{"source_id":"s1","title":"Test Video A"}],"context":[{"chunk_id":"c1","citation_index":1,"source_id":"s1","source_title":"Test Video A","timestamp_start":0,"timestamp_end":132,"rerank_score":4.53,"preview":"test preview"}],"reranked":true,"scoped_pool_size":10}}\n\n',
          'data: {"type":"delta","content":"Answer"}\n\n',
          'data: {"type":"done"}\n\n',
        ]),
      ),
    );

    renderChatPanel({
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hi");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => screen.getByText("Answer"));
    await waitFor(() => screen.getByText(/长期情景记忆/));
    // Summary line shows source count only (chunks moved to expanded detail)
    expect(screen.getByText(/1 sources/)).toBeInTheDocument();
  });

  test("two parallel find_passages calls produce two pending rows replaced independently", async () => {
    vi.spyOn(window, "fetch").mockImplementation(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"tool_call_start","id":"tc1","name":"find_passages","arguments":{"query":"A"}}\n\n',
          'data: {"type":"tool_call_start","id":"tc2","name":"find_passages","arguments":{"query":"B"}}\n\n',
          'data: {"type":"tool_result","id":"tc1","name":"find_passages","result":{"query":"A","tool_name":"find_passages","candidates_evaluated":10,"sources_with_hits":2,"sources_total":4,"source_coverage":[{"source_id":"s1","title":"Video A"},{"source_id":"s2","title":"Video B"}],"context":[{"chunk_id":"c1","citation_index":1,"source_id":"s1","source_title":"Video A","timestamp_start":0,"timestamp_end":60,"rerank_score":3.5,"preview":"preview A"}],"reranked":true,"scoped_pool_size":4}}\n\n',
          'data: {"type":"tool_result","id":"tc2","name":"find_passages","result":{"query":"B","tool_name":"find_passages","candidates_evaluated":20,"sources_with_hits":1,"sources_total":3,"source_coverage":[{"source_id":"s3","title":"Video C"}],"context":[{"chunk_id":"c2","citation_index":2,"source_id":"s3","source_title":"Video C","timestamp_start":10,"timestamp_end":90,"rerank_score":2.8,"preview":"preview B"}],"reranked":true,"scoped_pool_size":2}}\n\n',
          'data: {"type":"delta","content":"Answer"}\n\n',
          'data: {"type":"done"}\n\n',
        ]),
      ),
    );

    renderChatPanel({
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hi");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => screen.getByText("Answer"));

    expect(screen.getByText("A")).toBeInTheDocument();
    expect(screen.getByText("B")).toBeInTheDocument();
  });

});
