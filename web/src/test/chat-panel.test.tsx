import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
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

const SOURCE_2: Source = {
  id: "src-2",
  video_id: "BV1test2",
  platform: "bilibili",
  title: "Test Video B",
  summary: "Another test video",
  keywords: [],
  cover_url: null,
  source_url: "https://bilibili.com/video/BV1test2",
  duration_seconds: 1800,
  uploader: "TestUploader",
  language: "en",
  processed_at: "2026-04-08T13:00:00Z",
};

function makeConversationMock() {
  return vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
    const method = init?.method ?? "GET";
    if (url.includes("/conversation") && method === "GET") {
      return Promise.resolve(
        new Response(JSON.stringify({ conversation: null, messages: [] })),
      );
    }
    if (url.includes("/chat") && method === "POST") {
      return Promise.resolve(
        new Response(
          new ReadableStream({
            start(c) {
              c.enqueue(new TextEncoder().encode('data: {"type":"done"}\n\n'));
              c.close();
            },
          }),
          { headers: { "Content-Type": "text/event-stream" } },
        ),
      );
    }
    return Promise.resolve(new Response(JSON.stringify([])));
  });
}

function renderChatPanel(props?: Partial<React.ComponentProps<typeof ChatPanel>>) {
  makeConversationMock();
  return render(
    <LanguageProvider>
      <JobActivityProvider>
        <ChatPanel
          selectedSourceIds={[]}
          sources={[]}
          listId="list-1"
          onArtifactGenerated={vi.fn()}
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

describe("chat panel", () => {
  test("shows 'Nothing selected' empty state when no sources selected", () => {
    renderChatPanel({ selectedSourceIds: [], sources: [] });

    expect(screen.getByText("Nothing selected")).toBeInTheDocument();
    expect(screen.getByText(/select sources in the left panel/i)).toBeInTheDocument();
    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeDisabled();
    expect(screen.getByPlaceholderText(/select sources to start chatting/i)).toBeInTheDocument();
  });

  test("shows 'Ask your sources' empty state when sources selected but no conversation", async () => {
    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });

    await waitFor(() => {
      expect(screen.getByText("Ask your sources")).toBeInTheDocument();
    });
    expect(screen.getByText(/questions are answered from the transcripts/i)).toBeInTheDocument();
  });

  test("header shows source count and total duration", () => {
    renderChatPanel({ selectedSourceIds: ["src-1", "src-2"], sources: [SOURCE_1, SOURCE_2] });

    expect(screen.getByText("2 sources · 1h 30m total")).toBeInTheDocument();
  });

  test("clear button is disabled when no conversation exists", () => {
    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });

    const clearBtn = screen.getByRole("button", { name: /clear conversation/i });
    expect(clearBtn).toBeDisabled();
  });

  test("send button is disabled when input is empty", () => {
    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });

    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    expect(sendBtn).toBeDisabled();
  });

  test("send button is enabled when textarea has content", async () => {
    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hello");
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    expect(sendBtn).not.toBeDisabled();
  });

  test("Enter submits message, Shift+Enter creates newline", async () => {
    vi.spyOn(window, "fetch").mockImplementation(() =>
      Promise.resolve(
        new Response(
          new ReadableStream({
            start(c) {
              c.enqueue(new TextEncoder().encode('data: {"type":"done"}\n\n'));
              c.close();
            },
          }),
          { headers: { "Content-Type": "text/event-stream" } },
        ),
      ),
    );

    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hello world");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Hello world")).toBeInTheDocument();
    });
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    expect(sendBtn).toBeDisabled();
    expect(textarea).toHaveValue("");
  });

  test("Shift+Enter inserts newline without submitting", async () => {
    vi.spyOn(window, "fetch").mockImplementation(() =>
      Promise.resolve(
        new Response(
          new ReadableStream({
            start(c) {
              c.enqueue(new TextEncoder().encode('data: {"type":"done"}\n\n'));
              c.close();
            },
          }),
          { headers: { "Content-Type": "text/event-stream" } },
        ),
      ),
    );

    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Line 1{Shift>}{Enter}{/Shift}Line 2");
    // Textarea should still have the newline text (not submitted)
    expect(textarea).toHaveValue("Line 1\nLine 2");
    // Now submit with Enter
    await userEvent.keyboard("{Enter}");
    // Textarea should clear after submit
    await waitFor(() => {
      expect(textarea).toHaveValue("");
    });
  });

  test("textarea placeholder reflects state", () => {
    renderChatPanel({ selectedSourceIds: [], sources: [] });
    expect(screen.getByPlaceholderText(/select sources to start chatting/i)).toBeInTheDocument();

    cleanup();

    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });
    expect(screen.getByPlaceholderText(/ask about your sources/i)).toBeInTheDocument();
  });
});
