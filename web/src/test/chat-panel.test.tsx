import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ChatPanel } from "@/components/lists/ChatPanel";
import { TEST_IDS } from "@/lib/test-ids";
import type { Source } from "@/lib/types";
import { renderWithProviders } from "@/test/utils";

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

function renderChatPanel(
  props?: Partial<React.ComponentProps<typeof ChatPanel>>,
  { skipMock = false }: { skipMock?: boolean } = {},
) {
  if (!skipMock) makeConversationMock();
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
  vi.restoreAllMocks();
});

describe("chat panel", () => {
  test("citation chip renders cite-missing for unknown source", async () => {
    // Mock conversation with a citation block for a source not in the list
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
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
                      { type: "citation", index: 1, source_id: "deleted-source", chunk_ids: [] },
                    ],
                  },
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      expect(screen.getByText("[1]")).toBeInTheDocument();
    });
    // cite-missing span renders [1] with the i18n tooltip
    const chip = screen.getByText("[1]");
    expect(chip.getAttribute("data-testid")).toBe(TEST_IDS.citeMissing);
  });

  test("multi-paragraph response renders separate <p> elements", async () => {
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
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
                      { type: "text", text: "Para one " },
                      { type: "citation", index: 1, source_id: "src-1", chunk_ids: [] },
                      { type: "paragraph_break" },
                      { type: "text", text: "Para two." },
                    ],
                  },
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      const paras = document.querySelectorAll(".citation-paragraph");
      expect(paras.length).toBe(2);
    });
  });

  test("citation between two text fragments renders inline within one <p>", async () => {
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
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
                      { type: "text", text: "before " },
                      { type: "citation", index: 1, source_id: "src-1", chunk_ids: [] },
                      { type: "text", text: " after" },
                    ],
                  },
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      const paras = document.querySelectorAll(".citation-paragraph");
      expect(paras.length).toBe(1);
      expect(paras[0].querySelector(`[data-testid='${TEST_IDS.citeChip}']`)).not.toBeNull();
    });
  });

  test("citation after a lone paragraph_break attaches to the previous paragraph", async () => {
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
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
                      { type: "text", text: "a" },
                      { type: "paragraph_break" },
                      { type: "citation", index: 1, source_id: "src-1", chunk_ids: [] },
                    ],
                  },
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      const paras = document.querySelectorAll(".citation-paragraph");
      // Exactly one paragraph: "a" with the chip inline. No citation-only para.
      expect(paras.length).toBe(1);
      expect(paras[0].querySelector(`[data-testid='${TEST_IDS.citeChip}']`)).not.toBeNull();
      expect(paras[0].textContent).toContain("a");
      for (const p of paras) {
        const hasChip = p.querySelector(`[data-testid='${TEST_IDS.citeChip}']`) !== null;
        const text = (p.textContent ?? "").replace(/\[\d+\]/g, "").trim();
        expect(hasChip && text === "").toBe(false);
      }
    });
  });

  test("citation inside bullet list renders chip inline within <li>", async () => {
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
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
                      { type: "text", text: "- first point" },
                      { type: "citation", index: 1, source_id: "src-1", chunk_ids: [] },
                      { type: "text", text: "\n- second point" },
                      { type: "citation", index: 1, source_id: "src-1", chunk_ids: [] },
                    ],
                  },
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      // Exactly one <ul> — not two separate lists
      const uls = document.querySelectorAll("ul");
      expect(uls.length).toBe(1);
      // Both chips are inside <li> elements
      const chipsInLi = document.querySelectorAll(`li [data-testid='${TEST_IDS.citeChip}']`);
      expect(chipsInLi.length).toBe(2);
    });
  });

  test("citation after heading keeps chip inline within the heading block", async () => {
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
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
                      { type: "text", text: "## The Title" },
                      { type: "citation", index: 1, source_id: "src-1", chunk_ids: [] },
                      { type: "text", text: " trailing text" },
                    ],
                  },
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      // Chip is inside the same paragraph as the heading
      const headingPara = document.querySelector(".citation-paragraph:has(h2)");
      expect(headingPara).not.toBeNull();
      expect(headingPara!.querySelector(`[data-testid='${TEST_IDS.citeChip}']`)).not.toBeNull();
      const h2 = headingPara!.querySelector("h2");
      expect(h2).not.toBeNull();
      expect(h2!.textContent).toContain("The Title");
    });
  });

  test("consecutive citations in same paragraph both render as chips", async () => {
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
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
                      { type: "text", text: "Sources " },
                      { type: "citation", index: 1, source_id: "src-1", chunk_ids: [] },
                      { type: "citation", index: 2, source_id: "src-2", chunk_ids: [] },
                      { type: "text", text: " agree." },
                    ],
                  },
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1", "src-2"], sources: [SOURCE_1, SOURCE_2] },
      { skipMock: true },
    );

    await waitFor(() => {
      const chips = document.querySelectorAll(`[data-testid='${TEST_IDS.citeChip}']`);
      expect(chips.length).toBe(2);
      // Both chips in same paragraph
      const para = document.querySelector(".citation-paragraph");
      expect(para).not.toBeNull();
      expect(para!.querySelectorAll(`[data-testid='${TEST_IDS.citeChip}']`).length).toBe(2);
    });
  });

  test("CITE_TOKEN_RE split handles edge cases", async () => {
    const { CITE_TOKEN_RE: RE } = await import("@/components/lists/ChatPanel");

    // Adjacent tokens
    expect("​⁣CITE0⁣​​⁣CITE1⁣​".split(RE)).toEqual(["", "0", "", "1", ""]);

    // Token at start
    expect("​⁣CITE0⁣​text".split(RE)).toEqual(["", "0", "text"]);

    // Token at end
    expect("text​⁣CITE0⁣​".split(RE)).toEqual(["text", "0", ""]);

    // Token-only string
    expect("​⁣CITE3⁣​".split(RE)).toEqual(["", "3", ""]);

    // No token
    expect("plain text".split(RE)).toEqual(["plain text"]);

    // Text with token in middle
    expect("before ​⁣CITE2⁣​ after".split(RE)).toEqual(["before ", "2", " after"]);
  });

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

  test("known error code displays localized message", async () => {
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              conversation: { id: "conv-1", list_id: "list-1" },
              messages: [
                {
                  id: "msg-1",
                  role: "user",
                  content: "Hello",
                  metadata: null,
                  created_at: "2026-04-01T10:00:00Z",
                },
                {
                  id: "msg-2",
                  role: "assistant",
                  content: "",
                  error: "llm_rate_limit_error",
                  metadata: null,
                  created_at: "2026-04-01T10:01:00Z",
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      expect(screen.getByText("AI provider rate limit exceeded")).toBeInTheDocument();
    });
  });

  test("unknown error string displays as-is without i18n key match", async () => {
    vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              conversation: { id: "conv-1", list_id: "list-1" },
              messages: [
                {
                  id: "msg-1",
                  role: "user",
                  content: "Hello",
                  metadata: null,
                  created_at: "2026-04-01T10:00:00Z",
                },
                {
                  id: "msg-2",
                  role: "assistant",
                  content: "",
                  error: "An internal error occurred",
                  metadata: null,
                  created_at: "2026-04-01T10:01:00Z",
                },
              ],
            }),
          ),
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

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      expect(screen.getByText("An internal error occurred")).toBeInTheDocument();
    });
  });
});
