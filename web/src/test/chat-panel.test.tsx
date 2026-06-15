import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useState } from "react";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ChatPanel } from "@/components/lists/ChatPanel";
import { TEST_IDS } from "@/lib/test-ids";
import {
  makeSseStream,
  mockFetch,
  renderWithProviders,
  SOURCE_1,
  SOURCE_2,
} from "@/test/utils";

function makeConversationMock() {
  return mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (url.includes("/conversation") && method === "GET") {
      return Promise.resolve(
        new Response(JSON.stringify({ conversation: null, messages: [] })),
      );
    }
    if (url.includes("/chat") && method === "POST") {
      return Promise.resolve(
        makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
          makeSseStream(['data: {"type":"done"}\n\n'])
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

  test("shows </> debug button only when debug_prompts is on AND has_dump is true", async () => {
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
                  content: "Hello",
                  has_dump: true,
                  metadata: null,
                  created_at: "2026-04-01T10:01:00Z",
                },
              ],
            }),
          ),
        );
      }
      if (url.endsWith("/api/config") && method === "GET") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              rag: {
                max_distance: 0.7,
                reranking_enabled: true,
                hybrid_enabled: true,
                debug_prompts: true,
              },
            }),
          ),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        return Promise.resolve(makeSseStream(['data: {"type":"done"}\n\n']));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      expect(screen.getByTitle(/view llm context/i)).toBeInTheDocument();
    });
  });

  test("hides </> debug button when has_dump is false", async () => {
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
                  content: "Hello",
                  has_dump: false,
                  metadata: null,
                  created_at: "2026-04-01T10:01:00Z",
                },
              ],
            }),
          ),
        );
      }
      if (url.endsWith("/api/config") && method === "GET") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              rag: {
                max_distance: 0.7,
                reranking_enabled: true,
                hybrid_enabled: true,
                debug_prompts: true,
              },
            }),
          ),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        return Promise.resolve(makeSseStream(['data: {"type":"done"}\n\n']));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      expect(screen.getByText("Hello")).toBeInTheDocument();
    });
    expect(screen.queryByTitle(/view llm context/i)).toBeNull();
  });

  test("hides </> debug button when debug_prompts is off", async () => {
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
                  content: "Hello",
                  has_dump: true,
                  metadata: null,
                  created_at: "2026-04-01T10:01:00Z",
                },
              ],
            }),
          ),
        );
      }
      if (url.endsWith("/api/config") && method === "GET") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              rag: {
                max_distance: 0.7,
                reranking_enabled: true,
                hybrid_enabled: true,
                debug_prompts: false,
              },
            }),
          ),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        return Promise.resolve(makeSseStream(['data: {"type":"done"}\n\n']));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    await waitFor(() => {
      expect(screen.getByText("Hello")).toBeInTheDocument();
    });
    expect(screen.queryByTitle(/view llm context/i)).toBeNull();
  });

  test("opens drawer on icon click and closes on Esc", async () => {
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
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
                  content: "Hello",
                  has_dump: true,
                  metadata: null,
                  created_at: "2026-04-01T10:01:00Z",
                },
              ],
            }),
          ),
        );
      }
      if (url.endsWith("/api/config") && method === "GET") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              rag: {
                max_distance: 0.7,
                reranking_enabled: true,
                hybrid_enabled: true,
                debug_prompts: true,
              },
            }),
          ),
        );
      }
      if (url.includes("/debug/messages/") && method === "GET") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              system: "sys",
              tools: [],
              messages: [],
              response: { text: "final" },
              model: "test-model",
              timestamp: "2026-06-06T00:00:00Z",
            }),
          ),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        return Promise.resolve(makeSseStream(['data: {"type":"done"}\n\n']));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    const user = userEvent.setup();
    renderChatPanel(
      { selectedSourceIds: ["src-1"], sources: [SOURCE_1] },
      { skipMock: true },
    );

    const btn = await waitFor(() => screen.getByTitle(/view llm context/i));
    await user.click(btn);
    await waitFor(() => {
      expect(screen.getByTestId(TEST_IDS.debugDrawer)).toBeInTheDocument();
    });
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => {
      expect(screen.queryByTestId(TEST_IDS.debugDrawer)).toBeNull();
    });
  });

  test("pendingMessage auto-sends and consumes the prop", async () => {
    let postedMessage: string | null = null;
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
        return Promise.resolve(
          new Response(JSON.stringify({ conversation: null, messages: [] })),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        postedMessage = JSON.parse(String(init?.body ?? "{}")).message;
        return Promise.resolve(makeSseStream(['data: {"type":"done"}\n\n']));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    const onPendingMessageConsumed = vi.fn();
    renderChatPanel(
      {
        selectedSourceIds: ["src-1"],
        sources: [SOURCE_1],
        pendingMessage: { text: "Discuss alpha", nonce: 1 },
        onPendingMessageConsumed,
      },
      { skipMock: true },
    );

    await waitFor(() => {
      expect(postedMessage).toBe("Discuss alpha");
      expect(onPendingMessageConsumed).toHaveBeenCalledTimes(1);
    });
  });

  test("pendingMessage does not send when no sources are selected", async () => {
    let posted = false;
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
        return Promise.resolve(
          new Response(JSON.stringify({ conversation: null, messages: [] })),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        posted = true;
        return Promise.resolve(makeSseStream(['data: {"type":"done"}\n\n']));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    const onPendingMessageConsumed = vi.fn();
    renderChatPanel(
      {
        selectedSourceIds: [],
        sources: [],
        pendingMessage: { text: "Discuss alpha", nonce: 1 },
        onPendingMessageConsumed,
      },
      { skipMock: true },
    );

    // Give any stray effect a chance to fire.
    await new Promise((r) => setTimeout(r, 50));
    expect(posted).toBe(false);
    expect(onPendingMessageConsumed).not.toHaveBeenCalled();
  });

  test("new pendingMessage nonce re-fires send even with identical text", async () => {
    const postedMessages: string[] = [];
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
        return Promise.resolve(
          new Response(JSON.stringify({ conversation: null, messages: [] })),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        postedMessages.push(JSON.parse(String(init?.body ?? "{}")).message);
        return Promise.resolve(makeSseStream(['data: {"type":"done"}\n\n']));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    // Use a stateful wrapper so onPendingMessageConsumed can clear the
    // prop via a real setState (mirroring ListDetailPage). A bare
    // `vi.fn()` callback would never clear the prop, and the auto-send
    // effect would re-fire on every `isStreaming` flip.
    function Harness() {
      const [msg, setMsg] = useState<{ text: string; nonce: number } | null>({ text: "Discuss alpha", nonce: 1 });
      return (
        <>
          <button
            type="button"
            data-testid="set-msg-2"
            onClick={() => setMsg({ text: "Discuss alpha", nonce: 2 })}
          >
            set
          </button>
          <ChatPanel
            selectedSourceIds={["src-1"]}
            sources={[SOURCE_1]}
            listId="list-1"
            pendingMessage={msg}
            onPendingMessageConsumed={() => setMsg(null)}
          />
        </>
      );
    }

    renderWithProviders(<Harness />, { providers: [LanguageProvider, JobActivityProvider] });

    await waitFor(() => {
      expect(postedMessages).toEqual(["Discuss alpha"]);
    });

    fireEvent.click(screen.getByTestId("set-msg-2"));

    await waitFor(() => {
      expect(postedMessages).toEqual(["Discuss alpha", "Discuss alpha"]);
    });
  });
});
