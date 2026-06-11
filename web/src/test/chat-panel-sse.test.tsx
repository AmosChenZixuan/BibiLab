import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ChatPanel } from "@/components/lists/ChatPanel";
import { TEST_IDS } from "@/lib/test-ids";
import {
  makeOpenSseStream,
  makeSseStream,
  mockFetch,
  renderChatPanel,
  SOURCE_1,
} from "@/test/utils";

const ASSISTANT_MSG_ID = "msg-assistant-1";
const USER_MSG_ID = "msg-user-1";

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("chat panel — SSE streaming", () => {
  test("user message appears immediately, assistant streams in via SSE", async () => {
    const fetchSpy = mockFetch(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"delta","content":"Hello"}\n\n',
          'data: {"type":"delta","content":" world"}\n\n',
          'data: {"type":"done"}\n\n',
        ]),
      ),
    );

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hi there");
    await userEvent.keyboard("{Enter}");

    const userBubble = await waitFor(() => screen.getByText("Hi there"));
    expect(userBubble).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Hello world")).toBeInTheDocument();
    });

    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining("/api/lists/list-1/chat"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  test("pending find_passages ledger shows while streaming before any content", async () => {
    // find_passages dispatches mid-stream before any preamble text. The pending
    // ledger row must be visible even though the assistant bubble is empty
    // and the message is still streaming.
    const { response, enqueue } = makeOpenSseStream();
    mockFetch((input) => {
      const url = String(input);
      if (url.includes("/chat")) {
        return Promise.resolve(response);
      }
      // conversation GET: empty history
      return Promise.resolve(makeSseStream([]));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "What is X?");
    await userEvent.keyboard("{Enter}");

    enqueue(
      'data: {"type":"tool_call_start","id":"t1","name":"find_passages","arguments":{"query":"X"}}\n\n',
    );

    await waitFor(() => {
      expect(screen.getByText(/finding passages…/i)).toBeInTheDocument();
    });
  });

  test("ledger collapses mid-stream, becomes expandable with context after rag+done", async () => {
    const { response, enqueue, close } = makeOpenSseStream();
    mockFetch((input) => {
      const url = String(input);
      if (url.includes("/chat")) {
        return Promise.resolve(response);
      }
      return Promise.resolve(makeSseStream([]));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider], selectedSourceIds: ["src-1"], sources: [SOURCE_1], listId: "list-1" });

    await userEvent.type(screen.getByRole("textbox"), "what is X?");
    await userEvent.keyboard("{Enter}");

    enqueue('data: {"type":"meta","message_id":"srv-1"}\n\n');
    enqueue(
      'data: {"type":"tool_result","id":"r1","name":"find_passages","result":{"query":"X","tool_name":"find_passages","candidates_evaluated":5,"sources_with_hits":1,"sources_total":1,"section_coverage":[{"section_id":"sec-1","source_id":"s1","source_title":"Vid","seq":1,"timestamp_start":0,"timestamp_end":10}],"reranked":true,"scoped_pool_size":1}}\n\n',
    );

    // Mid-stream: ledger present but NOT expandable.
    await waitFor(() => {
      expect(screen.getByText(/X/)).toBeInTheDocument();
    });
    expect(
      document.querySelector('button[aria-label="Toggle retrieval details"]'),
    ).toBeNull();

    // Final authoritative rag (with context[]) then done.
    enqueue(
      'data: {"type":"rag","calls":[{"query":"X","tool_name":"find_passages","candidates_evaluated":5,"sources_with_hits":1,"sources_total":1,"section_coverage":[{"section_id":"sec-1","source_id":"s1","source_title":"Vid","seq":1,"timestamp_start":0,"timestamp_end":10}],"context":[{"chunk_id":"c1","citation_index":1,"section_id":"sec-1","section_seq":1,"source_id":"s1","source_title":"Vid","timestamp_start":0,"timestamp_end":10,"rerank_score":2.5,"preview":"unique-preview-text"}],"reranked":true,"scoped_pool_size":1}]}\n\n',
    );
    enqueue('data: {"type":"delta","content":"answer"}\n\n');
    enqueue('data: {"type":"done"}\n\n');
    close();

    const toggle = await waitFor(() => {
      const b = document.querySelector('button[aria-label="Toggle retrieval details"]');
      expect(b).not.toBeNull();
      return b as HTMLElement;
    });
    await userEvent.click(toggle);
    expect(screen.getByText(/unique-preview-text/)).toBeInTheDocument();
  });

  test("streaming delivers content to assistant bubble", async () => {
    mockFetch(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"delta","content":"Slow answer"}\n\n',
          'data: {"type":"done"}\n\n',
        ]),
      ),
    );

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Tell me something");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Slow answer")).toBeInTheDocument();
    });
  });

  test("error event shows error message + Retry button", async () => {
    mockFetch(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"delta","content":"Partial "}\n\n',
          'data: {"type":"error","message":"Connection closed"}\n\n',
        ]),
      ),
    );

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Tell me everything");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Partial")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("Connection closed")).toBeInTheDocument();
    });

    const retryBtn = screen.getByRole("button", { name: /retry/i });
    expect(retryBtn).toBeInTheDocument();
  });

  test("streamed citation after a \\n\\n renders inline, not on its own line", async () => {
    mockFetch((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/conversation")) {
        return Promise.resolve(new Response(JSON.stringify({ conversation: null, messages: [] })));
      }
      if (url.includes("/chat")) {
        return Promise.resolve(
          makeSseStream([
            'data: {"type":"delta","content":"出香味\\n\\n"}\n\n',
            'data: {"type":"citation","index":1,"source_id":"src-1","chunk_ids":[]}\n\n',
            'data: {"type":"done"}\n\n',
          ]),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "How to cook");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      const paras = document.querySelectorAll(".citation-paragraph");
      expect(paras.length).toBe(1);
      expect(paras[0].querySelector(`[data-testid='${TEST_IDS.citeChip}']`)).not.toBeNull();
      expect(paras[0].textContent).toContain("出香味");
      for (const p of paras) {
        const hasChip = p.querySelector(`[data-testid='${TEST_IDS.citeChip}']`) !== null;
        const text = (p.textContent ?? "").replace(/\[\d+\]/g, "").trim();
        expect(hasChip && text === "").toBe(false);
      }
    });
  });

  test("renders a read_section ledger chip when the tool fires mid-stream", async () => {
    const { response, enqueue, close } = makeOpenSseStream();
    mockFetch((input) => {
      const url = String(input);
      if (url.includes("/chat")) {
        return Promise.resolve(response);
      }
      return Promise.resolve(makeSseStream([]));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider], selectedSourceIds: ["src-1"], sources: [SOURCE_1], listId: "list-1" });

    await userEvent.type(screen.getByRole("textbox"), "read the section");
    await userEvent.keyboard("{Enter}");

    // Provisional read_section chip first (BookOpen + spinner + the
    // readSectionPending i18n text)
    enqueue(
      'data: {"type":"tool_call_start","id":"rs1","name":"read_section","arguments":{}}\n\n',
    );

    await waitFor(() => {
      expect(screen.getByText(/reading section/i)).toBeInTheDocument();
    });

    // tool_result resolves it to source_id/source_title
    enqueue(
      'data: {"type":"tool_result","id":"rs1","name":"read_section","result":{"tool_name":"read_section","source_id":"s1","source_title":"Ep 4","section_id":"sec-1","query":"","candidates_evaluated":0,"sources_with_hits":0,"sources_total":1,"section_coverage":[],"context":[],"reranked":false,"scoped_pool_size":1}}\n\n',
    );

    // Citation with section_id
    enqueue(
      'data: {"type":"citation","index":1,"section_id":"sec-1","source_id":"s1","timestamp_start":42,"chunk_ids":["c1"]}\n\n',
    );

    // Final rag event then done
    enqueue(
      'data: {"type":"rag","calls":[{"tool_name":"read_section","source_id":"s1","source_title":"Ep 4","section_id":"sec-1","query":"","candidates_evaluated":0,"sources_with_hits":0,"sources_total":1,"section_coverage":[],"context":[],"reranked":false,"scoped_pool_size":1}]}\n\n',
    );
    enqueue('data: {"type":"delta","content":"answer"}\n\n');
    enqueue('data: {"type":"done"}\n\n');
    close();

    // Resolved read_section row rendering is exercised in ToolLedgerRow
    // tests; here just verify the answer streams.
    await waitFor(() => {
      expect(screen.getByText("answer")).toBeInTheDocument();
    });
  });

  test("citation SSE event with numeric section_id is coerced to string before the chip click", async () => {
    // The backend's CitationRegistryEntry.section_id is the INTEGER
    // sections.id; the SSE event serializes it as a JSON number. The
    // FE's ContentBlock.citation declares section_id: string, and
    // SourcesViewerMode.resolveTargetIdx does strict equality against
    // SourceSection.section_id (also string). A number on one side
    // silently falls through to the timestampStart branch, landing
    // the reader on the wrong section. This test drives the
    // integration path (SSE event → useSSEStream → chat-utils.
    // coerceCitationEvent → CitationChip → onOpenSource) and asserts
    // the string arrives at the page-level handler.
    const onOpenSource = vi.fn();
    mockFetch((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/conversation")) {
        return Promise.resolve(new Response(JSON.stringify({ conversation: null, messages: [] })));
      }
      if (url.includes("/chat")) {
        return Promise.resolve(
          makeSseStream([
            'data: {"type":"delta","content":"see "}\n\n',
            'data: {"type":"citation","index":1,"section_id":1,"source_id":"src-1","timestamp_start":42,"chunk_ids":["c1"]}\n\n',
            'data: {"type":"delta","content":" [1] for the cited part."}\n\n',
            'data: {"type":"done"}\n\n',
          ]),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
      onOpenSource,
    });

    await userEvent.type(screen.getByRole("textbox"), "show");
    await userEvent.keyboard("{Enter}");

    const chip = await waitFor(() => screen.getByTestId(TEST_IDS.citeChip));
    await userEvent.click(chip);

    expect(onOpenSource).toHaveBeenCalledWith(
      expect.objectContaining({ id: "src-1" }),
      expect.objectContaining({
        sectionId: "1",
        timestampStart: 42,
      }),
    );
  });
});

describe("chat panel — conversation history", () => {
  test("on mount loads conversation via GET /lists/:id/conversation", async () => {
    const fetchSpy = mockFetch((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/conversation")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              conversation: { id: "conv-1", list_id: "list-1", summary: null, created_at: "2026-04-01T00:00:00Z", updated_at: "2026-04-01T00:00:00Z" },
              messages: [
                { id: USER_MSG_ID, role: "user", content: "What is backprop?", metadata: null, created_at: "2026-04-01T10:00:00Z" },
                { id: ASSISTANT_MSG_ID, role: "assistant", content: "It is the chain rule.", metadata: null, created_at: "2026-04-01T10:01:00Z" },
              ],
            }),
          ),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    await waitFor(() => {
      expect(screen.getByText("What is backprop?")).toBeInTheDocument();
    });
    expect(screen.getByText("It is the chain rule.")).toBeInTheDocument();

    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining("/api/lists/list-1/conversation"),
      expect.any(Object),
    );
  });

  test("clear button triggers DELETE /lists/:id/conversation and resets to empty state", async () => {
    let deleteCalled = false;
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "DELETE") {
        deleteCalled = true;
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      if (url.includes("/conversation")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              conversation: { id: "conv-1", list_id: "list-1", summary: null, created_at: "2026-04-01T00:00:00Z", updated_at: "2026-04-01T00:00:00Z" },
              messages: [
                { id: USER_MSG_ID, role: "user", content: "Hello", metadata: null, created_at: "2026-04-01T10:00:00Z" },
              ],
            }),
          ),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    await waitFor(() => screen.getByText("Hello"));

    await userEvent.click(screen.getByRole("button", { name: /clear conversation/i }));
    await userEvent.click(screen.getByRole("button", { name: /^clear$/i }));

    await waitFor(() => {
      expect(deleteCalled).toBe(true);
    });

    await waitFor(() => {
      expect(screen.getByText("Ask your sources")).toBeInTheDocument();
    });
  });

  test("message list dims to 50% opacity while clear popover is open", async () => {
    mockFetch((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/conversation")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              conversation: { id: "conv-1", list_id: "list-1", summary: null, created_at: "2026-04-01T00:00:00Z", updated_at: "2026-04-01T00:00:00Z" },
              messages: [
                { id: USER_MSG_ID, role: "user", content: "Hello", metadata: null, created_at: "2026-04-01T10:00:00Z" },
              ],
            }),
          ),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    await waitFor(() => screen.getByText("Hello"));

    await userEvent.click(screen.getByRole("button", { name: /clear conversation/i }));

    const messageList = screen.getByRole("region");
    expect(messageList.className).toMatch(/opacity-50/);
  });

  test("assistant message renders markdown as HTML (bold, code, lists)", async () => {
    mockFetch(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"delta","content":"The chain rule is **fundamental**.\\n\\nTwo components:\\n- Local gradient\\n- Upstream gradient\\n\\nCode: `x = 1`"}',
          '\n\n',
          'data: {"type":"done"}\n\n',
        ]),
      ),
    );

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Explain backprop");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      const strong = screen.getByText("fundamental");
      expect(strong.tagName).toBe("STRONG");

      const code = screen.getByText("x = 1");
      expect(code.tagName).toBe("CODE");

      expect(screen.getByText("Local gradient")).toBeInTheDocument();
    });
  });

  test("bubble uses bubble-user for user and bubble-assistant for assistant", async () => {
    mockFetch(() =>
      Promise.resolve(
        makeSseStream([
          'data: {"type":"delta","content":"Answer."}',
          '\n\n',
          'data: {"type":"done"}\n\n',
        ]),
      ),
    );

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hello");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      const userBubble = document.querySelector(`[data-testid='${TEST_IDS.bubbleUser}']`);
      expect(userBubble).not.toBeNull();
      expect(userBubble).toHaveTextContent("Hello");
    });

    await waitFor(() => {
      const assistantBubble = document.querySelector(`[data-testid='${TEST_IDS.bubbleAssistant}']`);
      expect(assistantBubble).not.toBeNull();
      expect(assistantBubble).toHaveTextContent("Answer.");
    });
  });

  test("retry on a mid-history failed assistant retries that specific turn's user message", async () => {
    let requestBody: string | null = null;
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
                  id: "user-1",
                  role: "user",
                  content: "First question",
                  metadata: null,
                  created_at: "2026-04-01T10:00:00Z",
                },
                {
                  id: "asst-1",
                  role: "assistant",
                  content: "",
                  error: "An internal error occurred",
                  metadata: null,
                  created_at: "2026-04-01T10:01:00Z",
                },
                {
                  id: "user-2",
                  role: "user",
                  content: "Second question",
                  metadata: null,
                  created_at: "2026-04-01T10:02:00Z",
                },
                {
                  id: "asst-2",
                  role: "assistant",
                  content: "Second answer",
                  metadata: null,
                  created_at: "2026-04-01T10:03:00Z",
                },
              ],
            }),
          ),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        const body = JSON.parse((init?.body as string) ?? "{}");
        requestBody = body.message;
        return Promise.resolve(
          makeSseStream(['data: {"type":"done"}\n\n']),
        );
      }
      if (url.includes("/cancel") && method === "POST") {
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    // Wait for history to load — two retry buttons should appear (one per failed assistant)
    const retryButtons = await waitFor(() => screen.getAllByRole("button", { name: /retry/i }));
    expect(retryButtons.length).toBeGreaterThanOrEqual(1);

    // Click retry on the first failed assistant (asst-1)
    await userEvent.click(retryButtons[0]);

    // Verify the correct user message text was re-sent (user-1 = "First question", not user-2)
    await waitFor(() => {
      expect(requestBody).toBe("First question");
    });

    // Subsequent messages after the retried turn are preserved (append-only retry)
    expect(screen.getByText("Second question")).toBeInTheDocument();
    expect(screen.getByText("Second answer")).toBeInTheDocument();
  });

  test("live SSE error with classified code displays localized message", async () => {
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url.includes("/conversation") && method === "GET") {
        return Promise.resolve(
          new Response(JSON.stringify({ conversation: null, messages: [] })),
        );
      }
      if (url.includes("/chat") && method === "POST") {
        return Promise.resolve(
          makeSseStream([
            'data: {"type":"error","message":"llm_rate_limit_error"}\n\n',
          ]),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderChatPanel(ChatPanel, { providers: [LanguageProvider, JobActivityProvider],
      selectedSourceIds: ["src-1"],
      sources: [SOURCE_1],
      listId: "list-1",
    });

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "test");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("AI provider rate limit exceeded")).toBeInTheDocument();
    });
  });
});
