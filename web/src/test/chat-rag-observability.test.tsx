import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ChatPanel } from "@/components/lists/ChatPanel";
import { TEST_IDS } from "@/lib/test-ids";
import { makeSseStream, mockFetch, renderWithProviders } from "@/test/utils";
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

function renderChatPanel(props?: Partial<React.ComponentProps<typeof ChatPanel>>) {
  return renderWithProviders(
    <ChatPanel
      selectedSourceIds={[]}
      sources={[]}
      listId="list-1"
      {...props}
    />,
    { providers: [LanguageProvider, JobActivityProvider] },
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("RAG observability via SSE tool_result", () => {
  test("tool_result followed by delta renders delta text in document", async () => {
    mockFetch(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"tool_result","name":"find_passages","result":{"query":"A","tool_name":"find_passages","candidates_evaluated":30,"sources_with_hits":1,"sources_total":1,"section_coverage":[{"section_id":"sec-1","source_id":"s1","source_title":"Test Video A","seq":1,"timestamp_start":0,"timestamp_end":132}],"context":[{"chunk_id":"c1","citation_index":1,"section_id":"sec-1","section_seq":1,"source_id":"s1","source_title":"Test Video A","timestamp_start":0,"timestamp_end":132,"rerank_score":4.53,"preview":"test preview"}],"reranked":true,"scoped_pool_size":1}}\n\n',
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
    mockFetch(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"tool_result","name":"find_passages","result":{"query":"长期情景记忆","tool_name":"find_passages","candidates_evaluated":10,"sources_with_hits":1,"sources_total":16,"section_coverage":[{"section_id":"sec-1","source_id":"s1","source_title":"Test Video A","seq":1,"timestamp_start":0,"timestamp_end":132}],"context":[{"chunk_id":"c1","citation_index":1,"section_id":"sec-1","section_seq":1,"source_id":"s1","source_title":"Test Video A","timestamp_start":0,"timestamp_end":132,"rerank_score":4.53,"preview":"test preview"}],"reranked":true,"scoped_pool_size":10}}\n\n',
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
    mockFetch(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"tool_call_start","id":"tc1","name":"find_passages","arguments":{"query":"A"}}\n\n',
          'data: {"type":"tool_call_start","id":"tc2","name":"find_passages","arguments":{"query":"B"}}\n\n',
          'data: {"type":"tool_result","id":"tc1","name":"find_passages","result":{"query":"A","tool_name":"find_passages","candidates_evaluated":10,"sources_with_hits":2,"sources_total":4,"section_coverage":[{"section_id":"sec-1","source_id":"s1","source_title":"Video A","seq":1,"timestamp_start":0,"timestamp_end":60},{"section_id":"sec-2","source_id":"s2","source_title":"Video B","seq":1,"timestamp_start":0,"timestamp_end":60}],"context":[{"chunk_id":"c1","citation_index":1,"section_id":"sec-1","section_seq":1,"source_id":"s1","source_title":"Video A","timestamp_start":0,"timestamp_end":60,"rerank_score":3.5,"preview":"preview A"}],"reranked":true,"scoped_pool_size":4}}\n\n',
          'data: {"type":"tool_result","id":"tc2","name":"find_passages","result":{"query":"B","tool_name":"find_passages","candidates_evaluated":20,"sources_with_hits":1,"sources_total":3,"section_coverage":[{"section_id":"sec-3","source_id":"s3","source_title":"Video C","seq":1,"timestamp_start":10,"timestamp_end":90}],"context":[{"chunk_id":"c2","citation_index":2,"section_id":"sec-3","section_seq":1,"source_id":"s3","source_title":"Video C","timestamp_start":10,"timestamp_end":90,"rerank_score":2.8,"preview":"preview B"}],"reranked":true,"scoped_pool_size":2}}\n\n',
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

  test("citation chip click passes section target to onOpenSource", async () => {
    const onOpenSource = vi.fn();
    mockFetch((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      if (url.includes("/conversation")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              conversation: { id: "conv-1", list_id: "list-1" },
              messages: [
                {
                  id: "msg-1",
                  role: "assistant",
                  content: "",
                  created_at: "2026-04-08T12:00:00Z",
                  metadata: {
                    content_blocks: [
                      { type: "text", text: "Answer " },
                      {
                        type: "citation",
                        index: 1,
                        section_id: "sec-1",
                        source_id: "src-1",
                        timestamp_start: 42,
                        chunk_ids: ["c1"],
                      },
                    ],
                  },
                },
              ],
            }),
          ),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel({
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
      onOpenSource,
    });

    const chip = await waitFor(() =>
      screen.getByTestId(TEST_IDS.citeChip),
    );
    await userEvent.click(chip);

    expect(onOpenSource).toHaveBeenCalledWith(
      expect.objectContaining({ id: "src-1" }),
      expect.objectContaining({
        highlightChunks: ["c1"],
        sectionId: "sec-1",
        timestampStart: 42,
      }),
    );
  });

});
