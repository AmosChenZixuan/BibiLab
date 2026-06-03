import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { Suspense } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider, useLanguage } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { routes } from "@/app/routes";

/** Wrap RouterProvider in Suspense so lazy routes load properly in tests */
function withRouter(router: ReturnType<typeof createMemoryRouter>) {
  return (
    <Suspense fallback={<div data-testid="router-loading">loading...</div>}>
      <JobActivityProvider>
        <LanguageProvider>
          <RouterProvider router={router} />
        </LanguageProvider>
      </JobActivityProvider>
    </Suspense>
  );
}

// ─── Per-test state (set inside each test) ───────────────────────────────────

const state = {
  sources: [] as Array<{
    id: string;
    video_id: string;
    platform: string;
    title: string;
    summary: string;
    keywords: string[];
    cover_url: string | null;
    source_url: string;
    duration_seconds: number;
    uploader: string;
    language: string | null;
    processed_at: string;
  }>,
  ingestCalls: [] as Array<{ listId: string; videos: unknown[] }>,
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
      return Promise.resolve(new Response(JSON.stringify({ queued: ["job-1"], skipped: [] })));
    }
    if (url.match(/\/api\/lists\/list-1\/sources\/source-old/) && method === "DELETE") {
      state.deletedIds.push("source-old");
      return Promise.resolve(new Response(null, { status: 204 }));
    }
    if (url.endsWith("/api/lists/list-1") && method === "PATCH") {
      return Promise.resolve(new Response(JSON.stringify({ ...state.renameResult })));
    }
    return Promise.resolve(new Response(JSON.stringify([])));
  });
}

// ─── API module mock ───────────────────────────────────────────────────────────

vi.mock("../lib/api", () => {
  const mockApi = {
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
    ingestUrl: vi.fn().mockImplementation((_listId: string, videos: unknown[]) => {
      state.ingestCalls.push({ listId: _listId, videos });
      return Promise.resolve({ queued: ["job-1"], skipped: [] });
    }),
    previewPlaylist: vi.fn().mockResolvedValue({ videos: [] }),
    previewPlaylistMetadata: vi.fn().mockResolvedValue({ videos: {} }),
    deleteSource: vi.fn().mockImplementation((_listId: string, sourceId: string) => {
      state.deletedIds.push(sourceId);
      return Promise.resolve();
    }),
    updateList: vi.fn().mockImplementation((_listId: string, _patch: object) =>
      Promise.resolve({ ...state.renameResult }),
    ),
    getSource: vi.fn().mockResolvedValue({
      id: "source-old",
      video_id: "BV1old",
      platform: "bilibili",
      title: "Existing Source",
      source_url: "https://www.bilibili.com/video/BV1old",
      duration_seconds: 600,
      uploader: "",
      language: null,
      processed_at: "2026-03-31T20:00:00Z",
      summary: "",
      keywords: [],
      cover_url: null,
      transcript: "This is the transcript text...",
      settings_snapshot: {},
    }),
    rerunDigest: vi.fn(),
    deleteJob: vi.fn(),
    createList: vi.fn(),
    putConfig: vi.fn(),
    listModels: vi.fn(),
    listArtifacts: vi.fn().mockResolvedValue([]),
    getConversation: vi.fn().mockResolvedValue({ conversation: null, messages: [] }),
    deleteConversation: vi.fn().mockResolvedValue(undefined),
  };
  return {
    HEALTH_REFRESH_EVENT: "bibilab:health:refresh",
    BILIBILI_AUTH_REFRESH_EVENT: "bibilab:auth:bilibili:refresh",
    JOBS_REFRESH_EVENT: "bibilab:jobs:refresh",
    notifyBilibiliAuthChanged: vi.fn(),
    createApiClient: () => mockApi,
    api: mockApi,
    toErrorMessageWithT: vi.fn((e: unknown) => (e instanceof Error ? e.message : String(e))),
    setCurrentLang: vi.fn(),
  };
});

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
    render(withRouter(router));

    expect(await screen.findByRole("heading", { name: /sources/i })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /lab/i })).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/select sources to start chatting/i)).toBeInTheDocument();
    expect(screen.getByText(/reports/i)).toBeInTheDocument();

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
    render(withRouter(router));

    let nav: HTMLElement | null = null;
    await waitFor(() => {
      nav = document.querySelector("nav") as HTMLElement | null;
      expect(nav).toBeInTheDocument();
    });
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
        id: "source-old",
        video_id: "BV1old",
        platform: "bilibili",
        title: "Existing Source",
        summary: "",
        keywords: [],
        cover_url: null,
        source_url: "https://www.bilibili.com/video/BV1old",
        duration_seconds: 0,
        uploader: "",
        language: null,
        processed_at: "2026-03-31T20:00:00Z",
      },
    ];
    makeMockFetch();
    vi.mocked(api.listSources).mockResolvedValue([...state.sources]);

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(withRouter(router));

    // Wait for sources to load before querying for source buttons
    await screen.findByRole("heading", { name: /sources/i });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /open existing source/i })).toBeInTheDocument();
    });

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    expect(input).toBeInTheDocument();

    vi.mocked(api.previewPlaylist).mockResolvedValue({
      videos: [
        {
          video_id: "BV1new",
          title: "New Video",
          cover_url: "https://example.com/cover.jpg",
          duration_seconds: 180,
          uploader: "New Author",
          platform: "bilibili",
          source_url: "https://www.bilibili.com/video/BV1new",
          part_label: null,
          status: "new" as const,
        },
      ],
    });

    await userEvent.type(input, "https://www.bilibili.com/video/BV1new");
    await userEvent.keyboard("{Enter}");
    expect(state.ingestCalls).toContainEqual({
      listId: "list-1",
      videos: expect.arrayContaining([
        expect.objectContaining({ video_id: "BV1new" }),
      ]),
    });
    expect(input).toHaveValue("");

    const sourceRow = screen
      .getByRole("button", { name: /open existing source/i })
      .closest("[class*='group']") as HTMLElement;
    await userEvent.hover(sourceRow);
    await userEvent.click(screen.getByRole("button", { name: /source options/i }));
    await userEvent.click(screen.getByText(/^delete$/i));
    expect(state.deletedIds).toContain("source-old");
    expect(screen.queryByRole("button", { name: /open existing source/i })).toBeNull();
  });


  test("opens viewer on source click, loads source content, shows Banner, DigestAccordion, and transcript", async () => {
    state.sources = [
      {
        id: "src-uuid",
        video_id: "BV1old",
        platform: "bilibili",
        title: "Existing Source",
        summary: "",
        keywords: [],
        cover_url: null,
        source_url: "https://www.bilibili.com/video/BV1old",
        duration_seconds: 0,
        uploader: "",
        language: null,
        processed_at: "2026-03-31T20:00:00Z",
      },
    ];
    makeMockFetch();
    vi.mocked(api.listSources).mockResolvedValue([...state.sources]);
    vi.mocked(api.getSource).mockResolvedValue({
      id: "src-uuid",
      video_id: "BV1old",
      platform: "bilibili",
      title: "Existing Source",
      source_url: "https://www.bilibili.com/video/BV1old",
      duration_seconds: 600,
      uploader: "TestUploader",
      language: null,
      processed_at: "2026-03-31T20:00:00Z",
      summary: "Great video",
      keywords: ["ml", "ai"],
      cover_url: null,
      transcript: "hello transcript",
      settings_snapshot: {},
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(withRouter(router));

    // Wait for sources to load before querying for source buttons
    await screen.findByRole("heading", { name: /sources/i });
    const sourceBtn = await waitFor(() => {
      return screen.getByRole("button", { name: /open existing source/i });
    });
    expect(sourceBtn).toBeInTheDocument();

    // 2. Clicking fires getSource

    // 2. Clicking fires getSource
    await userEvent.click(sourceBtn);
    expect(api.getSource).toHaveBeenCalledWith("src-uuid", expect.any(Object));

    // 3. Panel switches to viewer mode — source title in viewer header, source rows gone
    expect(await screen.findByText("Existing Source")).toBeInTheDocument(); // viewer header title
    expect(screen.queryByRole("button", { name: /open existing source/i })).not.toBeInTheDocument();

    // 4. Banner is present (cover image area)
    const banner = document.querySelector(".relative.h-64");
    expect(banner).toBeInTheDocument();

    // 5. DigestAccordion renders with "Digest" header (expanded by default)
    const digestAccordion = screen.getByText("Digest");
    expect(digestAccordion).toBeInTheDocument();

    // 6. No ReactMarkdown output (no .prose class)
    const proseElements = document.querySelectorAll(".prose");
    expect(proseElements.length).toBe(0);

    // 7. Keyword chips for "ml" and "ai" are visible after expanding DigestAccordion
    await userEvent.click(digestAccordion);
    expect(await screen.findByText("ml")).toBeInTheDocument();
    expect(screen.getByText("ai")).toBeInTheDocument();

    // 8. Transcript text "hello transcript" is present
    expect(screen.getByText("hello transcript")).toBeInTheDocument();

    // 9. Close button returns to list mode
    await userEvent.click(screen.getByRole("button", { name: /close viewer/i }));
    expect(screen.getByRole("button", { name: /open existing source/i })).toBeInTheDocument();
  });

  test("deselection persists after language switch", async () => {
    state.sources = [
      {
        id: "src-1",
        video_id: "BV1first",
        platform: "bilibili",
        title: "Source One",
        summary: "",
        keywords: [],
        cover_url: null,
        source_url: "https://www.bilibili.com/video/BV1first",
        duration_seconds: 0,
        uploader: "",
        language: null,
        processed_at: "2026-03-31T20:00:00Z",
      },
      {
        id: "src-2",
        video_id: "BV1second",
        platform: "bilibili",
        title: "Source Two",
        summary: "",
        keywords: [],
        cover_url: null,
        source_url: "https://www.bilibili.com/video/BV1second",
        duration_seconds: 0,
        uploader: "",
        language: null,
        processed_at: "2026-03-31T20:00:00Z",
      },
    ];
    makeMockFetch();
    vi.mocked(api.listSources).mockResolvedValue([...state.sources]);

    function LangToggle() {
      const { lang, setLang } = useLanguage();
      return (
        <button type="button" onClick={() => setLang(lang === "en" ? "zh" : "en")}>
          Toggle Lang ({lang})
        </button>
      );
    }

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(
      <Suspense fallback={<div data-testid="router-loading">loading...</div>}>
        <JobActivityProvider>
          <LanguageProvider>
            <RouterProvider router={router} />
            <LangToggle />
          </LanguageProvider>
        </JobActivityProvider>
      </Suspense>,
    );

    await screen.findByRole("heading", { name: /sources/i });

    const source1Checkbox = document.querySelectorAll('input[type="checkbox"]')[1];
    expect(source1Checkbox).toBeChecked();
    await userEvent.click(source1Checkbox);
    expect(source1Checkbox).not.toBeChecked();

    await userEvent.click(screen.getByRole("button", { name: /toggle lang/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /toggle lang \(zh\)/i })).toBeInTheDocument();
    });

    const source1CheckboxAfterLangSwitch = document.querySelectorAll('input[type="checkbox"]')[1];
    expect(source1CheckboxAfterLangSwitch).not.toBeChecked();
  });
});
