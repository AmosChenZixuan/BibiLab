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

describe("Slice 2 — RAG observability via SSE rag_meta", () => {
  test("rag_meta event attaches rag to in-progress message", async () => {
    vi.spyOn(window, "fetch").mockImplementation(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"rag_meta","rag":{"mode":"focused","candidates_evaluated":30,"sources_with_hits":1,"sources_total":1,"sources":[{"video_id":"BV1test","title":"Test Video A"}]}}\n\n',
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
      expect(screen.getByText(/30 chunks · 1\/1/)).toBeInTheDocument();
    });
  });

  test("Obs chip renders with correct chunk/source counts", async () => {
    vi.spyOn(window, "fetch").mockImplementation(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"rag_meta","rag":{"mode":"focused","candidates_evaluated":42,"sources_with_hits":3,"sources_total":5,"sources":[{"video_id":"v1","title":"Video A"},{"video_id":"v2","title":"Video B"},{"video_id":"v3","title":"Video C"}]}}\n\n',
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

    const chip = screen.getByText(/42 chunks · 3\/5/);
    expect(chip).toBeInTheDocument();
  });

  test("Obs chip expands on click", async () => {
    vi.spyOn(window, "fetch").mockImplementation(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"rag_meta","rag":{"mode":"focused","candidates_evaluated":10,"sources_with_hits":2,"sources_total":4,"sources":[{"video_id":"v1","title":"Intro Video"},{"video_id":"v2","title":"Conclusion Video"}]}}\n\n',
          'data: {"type":"delta","content":"Done"}\n\n',
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

    await waitFor(() => screen.getByText("Done"));

    const chip = screen.getByText(/10 chunks · 2\/4/);
    await userEvent.click(chip);

    await waitFor(() => {
      expect(screen.getByText(/focused/i)).toBeInTheDocument();
      expect(screen.getByText(/intro video/i)).toBeInTheDocument();
      expect(screen.getByText(/conclusion video/i)).toBeInTheDocument();
    });
  });
});
