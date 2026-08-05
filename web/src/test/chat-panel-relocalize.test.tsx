import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider, useLanguage } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ChatPanel } from "@/components/lists/ChatPanel";
import { makeSseStream, mockFetch, renderWithProviders, SOURCE_1 } from "@/test/utils";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
});

/** Language toggle rendered next to the panel — `setLang` is only reachable
 *  from inside the provider. */
function LangSwitch() {
  const { setLang } = useLanguage();
  return (
    <button type="button" onClick={() => setLang("zh")}>
      to-zh
    </button>
  );
}

/** One message per render path: user bubble, `cancelled` badge, `failed` badge,
 *  an `error` with no `chat.errors.*` key (the raw fallback), a clean assistant
 *  message (the only one reaching the assistant timestamp row), and one dated
 *  today — the only one that reads `chat.time.today`. */
const TODAY = new Date().toISOString();

const HISTORY = {
  conversation: { id: "conv-1", list_id: "list-1" },
  messages: [
    {
      id: "msg-0",
      role: "user",
      content: "a question",
      created_at: "2026-04-08T11:59:00Z",
    },
    {
      id: "msg-1",
      role: "assistant",
      content: "cancelled turn",
      created_at: "2026-04-08T12:00:00Z",
      status: "cancelled",
    },
    {
      id: "msg-2",
      role: "assistant",
      content: "failed turn",
      created_at: "2026-04-08T12:01:00Z",
      status: "failed",
    },
    {
      id: "msg-3",
      role: "assistant",
      content: "clean turn",
      created_at: "2026-04-08T12:02:00Z",
    },
    {
      id: "msg-4",
      role: "assistant",
      content: "rate limited turn",
      created_at: "2026-04-08T12:03:00Z",
      error: "rate limit exceeded",
    },
    {
      id: "msg-5",
      role: "assistant",
      content: "today turn",
      created_at: TODAY,
    },
  ],
};

describe("chat panel — language switch relocalizes loaded history", () => {
  test("badges and timestamps follow the language with no refetch", async () => {
    const fetchSpy = mockFetch((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/conversation")) {
        return Promise.resolve(new Response(JSON.stringify(HISTORY)));
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderWithProviders(
      <>
        <LangSwitch />
        <ChatPanel selectedSourceIds={["src-1"]} sources={[SOURCE_1]} listId="list-1" />
      </>,
      { providers: [LanguageProvider, JobActivityProvider] },
    );

    await waitFor(() => expect(screen.getByText("Stopped")).toBeInTheDocument());
    expect(screen.getByText("Response interrupted")).toBeInTheDocument();
    // No `chat.errors.*` key — getErrorLabel renders the string as-is.
    expect(screen.getByText("rate limit exceeded")).toBeInTheDocument();
    // Dated branch: en "Apr 8 HH:MM", zh "4月8日 HH:MM". Exactly two — the user
    // bubble and the clean assistant message are separate render sites, so a
    // count pins both; a regression on one cannot hide behind the other.
    expect(screen.getAllByText(/^Apr \d+ \d\d:\d\d$/)).toHaveLength(2);
    // todayLabel branch — the only assertion that reads `chat.time.today`.
    expect(screen.getByText(/^Today \d\d:\d\d$/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "to-zh" }));

    await waitFor(() => expect(screen.getByText("已停止")).toBeInTheDocument());
    expect(screen.getByText("回答被中断")).toBeInTheDocument();
    // The fallback string has no translation — it must survive the switch as-is.
    expect(screen.getByText("rate limit exceeded")).toBeInTheDocument();
    expect(screen.getAllByText(/^\d+月\d+日 \d\d:\d\d$/)).toHaveLength(2);
    expect(screen.getByText(/^今日 \d\d:\d\d$/)).toBeInTheDocument();
    expect(screen.queryByText(/Apr|Today/)).not.toBeInTheDocument();

    // Pins the rejected fix — labels in the dep array, relocalizing by refetch.
    // Not the stale-label guard: the old deps list did not refetch either.
    const conversationCalls = fetchSpy.mock.calls.filter(([input]) =>
      String(input).includes("/conversation"),
    ).length;
    expect(conversationCalls).toBe(1);
  });

  test("a cancelled stream stores the raw code, not a baked label", async () => {
    // The en badge alone cannot catch a regression here — a hardcoded "Stopped"
    // reads identically. Only the switched language proves the code was stored.
    mockFetch((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/conversation")) {
        return Promise.resolve(
          new Response(JSON.stringify({ conversation: null, messages: [] })),
        );
      }
      if (url.includes("/chat") && init?.method === "POST") {
        return Promise.resolve(
          makeSseStream([
            'data: {"type":"delta","content":"partial"}\n\n',
            'data: {"type":"cancelled"}\n\n',
          ]),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderWithProviders(
      <>
        <LangSwitch />
        <ChatPanel selectedSourceIds={["src-1"]} sources={[SOURCE_1]} listId="list-1" />
      </>,
      { providers: [LanguageProvider, JobActivityProvider] },
    );

    await userEvent.type(screen.getByRole("textbox"), "Hi");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => expect(screen.getByText("Stopped")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "to-zh" }));

    await waitFor(() => expect(screen.getByText("已停止")).toBeInTheDocument());
    expect(screen.queryByText("Stopped")).not.toBeInTheDocument();
  });

  test("a reattached message renders no timestamp rather than Invalid Date", async () => {
    // reattach() is the only producer of an empty createdAt; once its stream
    // ends the timestamp row mounts, so the render site must not format "".
    mockFetch((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/conversation")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              conversation: { id: "conv-1", list_id: "list-1", active_stream_message_id: "ghost-1" },
              messages: [],
            }),
          ),
        );
      }
      if (url.includes("/stream")) {
        return Promise.resolve(
          makeSseStream(['data: {"type":"delta","content":"resumed"}\n\n', 'data: {"type":"done"}\n\n']),
        );
      }
      return Promise.resolve(new Response(JSON.stringify([])));
    });

    renderWithProviders(
      <ChatPanel selectedSourceIds={["src-1"]} sources={[SOURCE_1]} listId="list-1" />,
      { providers: [LanguageProvider, JobActivityProvider] },
    );

    await waitFor(() => expect(screen.getByText("resumed")).toBeInTheDocument());
    expect(screen.queryByText(/Invalid Date/)).not.toBeInTheDocument();
  });
});
