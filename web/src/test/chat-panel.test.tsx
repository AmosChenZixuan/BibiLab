import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
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

function renderChatPanel(props?: Partial<React.ComponentProps<typeof ChatPanel>>) {
  return render(
    <LanguageProvider>
      <ChatPanel
        selectedSourceIds={[]}
        sources={[]}
        onSendMessage={vi.fn()}
        {...props}
      />
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

  test("shows 'Ask your sources' empty state + suggestion chips when sources selected but no conversation", () => {
    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });

    expect(screen.getByText("Ask your sources")).toBeInTheDocument();
    expect(screen.getByText(/questions are answered from the transcripts/i)).toBeInTheDocument();
    const chips = screen.getAllByRole("button", { name: /^→/ });
    expect(chips).toHaveLength(3);
  });

  test("clicking a suggestion chip calls onSendMessage with chip text", async () => {
    const onSendMessage = vi.fn();
    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1], onSendMessage });

    const chips = screen.getAllByRole("button", { name: /^→/ });
    await userEvent.click(chips[0]);

    expect(onSendMessage).toHaveBeenCalledTimes(1);
    expect(typeof onSendMessage.mock.calls[0][0]).toBe("string");
    expect(onSendMessage.mock.calls[0][0].length).toBeGreaterThan(0);
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
    const onSendMessage = vi.fn();
    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1], onSendMessage });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hello world");
    await userEvent.keyboard("{Enter}");

    expect(onSendMessage).toHaveBeenCalledWith("Hello world");
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    expect(sendBtn).toBeDisabled();
    expect(textarea).toHaveValue("");
  });

  test("Shift+Enter inserts newline without submitting", async () => {
    const onSendMessage = vi.fn();
    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1], onSendMessage });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Line 1{Shift>}{Enter}{/Shift}Line 2");
    await userEvent.keyboard("{Enter}");

    expect(onSendMessage).toHaveBeenCalledTimes(1);
    expect(onSendMessage).toHaveBeenCalledWith("Line 1\nLine 2");
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    expect(sendBtn).toBeDisabled();
  });

  test("textarea placeholder reflects state", () => {
    renderChatPanel({ selectedSourceIds: [], sources: [] });
    expect(screen.getByPlaceholderText(/select sources to start chatting/i)).toBeInTheDocument();

    cleanup();

    renderChatPanel({ selectedSourceIds: ["src-1"], sources: [SOURCE_1] });
    expect(screen.getByPlaceholderText(/ask about your sources/i)).toBeInTheDocument();
  });
});
