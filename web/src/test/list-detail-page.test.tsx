import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { routes } from "@/app/routes";

// ─── Per-test state (set inside each test) ───────────────────────────────────

const state = {
  sources: [] as Array<{
    video_id: string;
    platform: string;
    title: string;
    note_path: string;
    processed_at: string;
  }>,
  ingestCalls: [] as Array<{ url: string; rerun: boolean }>,
  deletedIds: [] as string[],
  renameResult: {
    id: "list-1",
    name: "New Name",
    created_at: "2026-03-31T19:00:00Z",
    thumbnail_source_id: null,
    thumbnail_url: null,
    source_count: 0,
    updated_at: "2026-03-31T21:00:00Z",
  },
};

// ─── Mock fetch helper ─────────────────────────────────────────────────────────

function makeMockFetch() {
  return vi.spyOn(window, "fetch").mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
    const method = init?.method ?? "GET";

    if (url.endsWith("/api/lists") && method === "GET") {
      return Promise.resolve(
        new Response(
          JSON.stringify([
            {
              id: "list-1",
              name: "AI Reading List",
              created_at: "2026-03-31T19:00:00Z",
              thumbnail_source_id: null,
              thumbnail_url: null,
              source_count: 0,
              updated_at: "2026-03-31T20:00:00Z",
            },
          ]),
        ),
      );
    }
    if (url.endsWith("/api/lists/list-1/sources") && method === "GET") {
      return Promise.resolve(new Response(JSON.stringify([...state.sources])));
    }
    if (url.endsWith("/api/jobs") && method === "GET") {
      return Promise.resolve(new Response(JSON.stringify([])));
    }
    if (url.includes("/api/ingest/url") && method === "POST") {
      const rerun = url.includes("rerun=true");
      const bodyStr = init?.body as string | undefined;
      let urlFromBody = "";
      if (bodyStr) {
        try {
          const body = JSON.parse(bodyStr);
          urlFromBody = body.url ?? "";
        } catch {}
      }
      state.ingestCalls.push({ url: urlFromBody, rerun });
      return Promise.resolve(new Response(JSON.stringify({ queued: ["job-1"], skipped: [] })));
    }
    if (url.match(/\/api\/lists\/list-1\/sources\/BV1old/) && method === "DELETE") {
      state.deletedIds.push("BV1old");
      return Promise.resolve(new Response(null, { status: 204 }));
    }
    if (url.endsWith("/api/lists/list-1") && method === "PATCH") {
      return Promise.resolve(new Response(JSON.stringify({ ...state.renameResult })));
    }
    return Promise.resolve(new Response(JSON.stringify([])));
  });
}

// ─── API module mock ───────────────────────────────────────────────────────────

vi.mock("../lib/api", () => ({
  HEALTH_REFRESH_EVENT: "locus:health:refresh",
  JOBS_REFRESH_EVENT: "locus:jobs:refresh",
  api: {
    getHealth: vi.fn().mockResolvedValue({
      overall: "ok",
      dependencies: {
        cuda: { status: "ok", message: "" },
        embedding_model: { status: "ok", message: "" },
      },
    }),
    listJobs: vi.fn().mockResolvedValue([]),
    listLists: vi.fn().mockResolvedValue([
      {
        id: "list-1",
        name: "AI Reading List",
        created_at: "2026-03-31T19:00:00Z",
        thumbnail_source_id: null,
        thumbnail_url: null,
        source_count: 0,
        updated_at: "2026-03-31T20:00:00Z",
      },
    ]),
    listSources: vi.fn().mockImplementation(() => Promise.resolve([...state.sources])),
    ingestUrl: vi.fn().mockImplementation((_listId: string, url: string, rerun: boolean) => {
      state.ingestCalls.push({ url, rerun });
      return Promise.resolve({ queued: ["job-1"], skipped: [] });
    }),
    deleteSource: vi.fn().mockImplementation((_listId: string, videoId: string) => {
      state.deletedIds.push(videoId);
      return Promise.resolve();
    }),
    updateList: vi.fn().mockImplementation((_listId: string, _patch: object) =>
      Promise.resolve({ ...state.renameResult }),
    ),
    getNoteContent: vi.fn().mockResolvedValue({
      markdown: "# Key Insights\n\nSome content here",
    }),
    getNoteTranscript: vi.fn().mockResolvedValue({
      transcript: "This is the transcript text...",
    }),
  },
  toErrorMessage: vi.fn((e: unknown) => (e instanceof Error ? e.message : String(e))),
}));

import { api } from "@/lib/api";

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  state.sources = [];
  state.ingestCalls = [];
  state.deletedIds = [];
  state.renameResult = {
    id: "list-1",
    name: "New Name",
    created_at: "2026-03-31T19:00:00Z",
    thumbnail_source_id: null,
    thumbnail_url: null,
    source_count: 0,
    updated_at: "2026-03-31T21:00:00Z",
  };
});

describe("list detail page", () => {
  test("renders three-panel workspace with correct panel headings and skeleton text", async () => {
    makeMockFetch();
    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    expect(screen.getByRole("heading", { name: /sources/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /lab/i })).toBeInTheDocument();
    expect(screen.getByText(/list-scoped chat arrives in v1/i)).toBeInTheDocument();
    expect(screen.getByText(/synthesis tools/i)).toBeInTheDocument();

    const collapseBtn = screen.getByRole("button", { name: /collapse sources/i });
    expect(collapseBtn).toBeInTheDocument();
    await userEvent.click(collapseBtn);
    expect(screen.getByRole("button", { name: /expand sources/i })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /expand sources/i }));
    expect(screen.getByRole("button", { name: /collapse sources/i })).toBeInTheDocument();
  });

  test("shows list name in navbar and commits rename on blur", async () => {
    makeMockFetch();
    vi.mocked(api.listSources).mockResolvedValue([]);
    vi.mocked(api.listJobs).mockResolvedValue([]);

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    const nav = document.querySelector("nav");
    expect(nav).toBeInTheDocument();
    expect(await screen.findByText("AI Reading List")).toBeInTheDocument();

    await userEvent.click(screen.getByText("AI Reading List"));
    const navWithin = within(nav!);
    const input = navWithin.getByRole("textbox");
    expect(input).toHaveValue("AI Reading List");

    await userEvent.clear(input);
    await userEvent.type(input, "New Name");
    await userEvent.tab();

    expect(await screen.findByText("New Name")).toBeInTheDocument();

    await userEvent.click(screen.getByText("New Name"));
    const navWithin2 = within(nav!);
    const input2 = navWithin2.getByRole("textbox");
    await userEvent.clear(input2);
    await userEvent.type(input2, "Should Not Commit");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByText("Should Not Commit")).not.toBeInTheDocument();
    expect(screen.getByText("New Name")).toBeInTheDocument();
  });

  test("loads sources, submits URL, cancels ingestion, and deletes source via context menu", async () => {
    state.sources = [
      {
        video_id: "BV1old",
        platform: "bilibili",
        title: "Existing Source",
        note_path: "/tmp/BV1old.md",
        processed_at: "2026-03-31T20:00:00Z",
      },
    ];
    makeMockFetch();
    vi.mocked(api.listSources).mockResolvedValue([...state.sources]);

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("button", { name: /open existing source/i })).toBeInTheDocument();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    expect(input).toBeInTheDocument();
    await userEvent.type(input, "https://www.bilibili.com/video/BV1new");
    await userEvent.keyboard("{Enter}");
    expect(state.ingestCalls).toContainEqual({ url: "https://www.bilibili.com/video/BV1new", rerun: false });
    expect(input).toHaveValue("");

    const sourceRow = screen
      .getByRole("button", { name: /open existing source/i })
      .closest("[class*='group']") as HTMLElement;
    await userEvent.hover(sourceRow);
    await userEvent.click(screen.getByRole("button", { name: /source options/i }));
    await userEvent.click(screen.getByText(/^delete$/i));
    expect(state.deletedIds).toContain("BV1old");
    expect(screen.queryByRole("button", { name: /open existing source/i })).toBeNull();
  });

  test("re-run context menu item calls ingestUrl with rerun=true", async () => {
    state.sources = [
      {
        video_id: "BV1old",
        platform: "bilibili",
        title: "Existing Source",
        note_path: "/tmp/BV1old.md",
        processed_at: "2026-03-31T20:00:00Z",
      },
    ];
    makeMockFetch();
    vi.mocked(api.listSources).mockResolvedValue([...state.sources]);

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("button", { name: /open existing source/i })).toBeInTheDocument();

    const sourceRow = screen
      .getByRole("button", { name: /open existing source/i })
      .closest("[class*='group']") as HTMLElement;
    await userEvent.hover(sourceRow);
    await userEvent.click(screen.getByRole("button", { name: /source options/i }));
    await userEvent.click(screen.getByText(/^re-run$/i));
    expect(state.ingestCalls).toContainEqual({
      url: "https://www.bilibili.com/video/BV1old",
      rerun: true,
    });
  });

  test("opens viewer on source click, lazy-loads transcript, close returns to list", async () => {
    state.sources = [
      {
        video_id: "BV1old",
        platform: "bilibili",
        title: "Existing Source",
        note_path: "/tmp/BV1old.md",
        processed_at: "2026-03-31T20:00:00Z",
      },
    ];
    makeMockFetch();
    vi.mocked(api.listSources).mockResolvedValue([...state.sources]);
    vi.mocked(api.getNoteContent).mockResolvedValue({
      video_id: "BV1old",
      title: "Existing Source",
      markdown: "# Key Insights\n\nSome content here",
    });
    vi.mocked(api.getNoteTranscript).mockResolvedValue({
      video_id: "BV1old",
      text: "This is the transcript text...",
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    // 1. Source row button is visible
    const sourceBtn = await screen.findByRole("button", { name: /open existing source/i });
    expect(sourceBtn).toBeInTheDocument();

    // 2. Clicking fires GET /api/notes/{video_id}/content
    await userEvent.click(sourceBtn);
    expect(api.getNoteContent).toHaveBeenCalledWith("BV1old");

    // 3. Panel switches to viewer mode — source title in viewer header, source rows gone
    expect(await screen.findByText("Existing Source")).toBeInTheDocument(); // viewer header title
    expect(screen.queryByRole("button", { name: /open existing source/i })).not.toBeInTheDocument();

    // 4. NoteAccordion renders with "Note" header (collapsed by default)
    const noteAccordion = screen.getByRole("button", { name: /^note$/i });
    expect(noteAccordion).toBeInTheDocument();
    // Note content is not yet visible (collapsed)
    expect(screen.queryByText(/Key Insights/i)).not.toBeInTheDocument();

    // 5. Clicking "Note" header expands it and shows markdown text
    await userEvent.click(noteAccordion);
    // Wait for the accordion to expand and show content
    await expect.poll(() => screen.queryByText(/Key Insights/i)).toBeInTheDocument();

    // 6. Transcript auto-loaded when viewer opened (no "Load transcript" button)
    expect(api.getNoteTranscript).toHaveBeenCalledWith("BV1old");
    // Transcript text appears directly (no button needed)
    expect(await screen.findByText("This is the transcript text...")).toBeInTheDocument();
    // "Transcript" label is visible
    expect(screen.getByText(/^transcript$/i)).toBeInTheDocument();

    // 8. Close button returns to list mode
    await userEvent.click(screen.getByRole("button", { name: /close viewer/i }));
    expect(screen.getByRole("button", { name: /open existing source/i })).toBeInTheDocument();
  });
});
