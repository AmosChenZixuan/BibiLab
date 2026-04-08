import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { routes } from "@/app/routes";

/** Wrap RouterProvider */
function withRouter(router: ReturnType<typeof createMemoryRouter>) {
  return (
    <JobActivityProvider>
      <LanguageProvider>
        <RouterProvider router={router} />
      </LanguageProvider>
    </JobActivityProvider>
  );
}

function installFetchMock(handler: (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>) {
  vi.stubGlobal("fetch", vi.fn(handler));
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("home page", () => {
  test("shows the my lists shell, creates an untitled list from the first tile, and deletes an existing list through a dialog", async () => {
    const lists = [
      {
        id: "list-1",
        name: "Systems",
        created_at: "2026-03-31T19:00:00Z",
        thumbnail_source_id: null,
        thumbnail_url: null,
        source_count: 0,
        updated_at: "2026-03-31T19:00:00Z",
      },
    ];

    installFetchMock(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url.endsWith("/api/health") && method === "GET") {
        return Response.json({
          overall: "ok",
          dependencies: {
            backend: { status: "ok", message: "" },
          },
        });
      }

      if (url.endsWith("/api/lists") && method === "GET") {
        return Response.json(lists);
      }

      if (url.endsWith("/api/lists") && method === "POST") {
        const body = JSON.parse(String(init?.body));
        expect(body).toEqual({ name: "Untitled list" });
        return Response.json({
          id: "list-2",
          name: body.name,
          created_at: "2026-03-31T19:01:00Z",
          thumbnail_source_id: null,
          thumbnail_url: null,
          source_count: 0,
          updated_at: "2026-03-31T19:01:00Z",
        });
      }

      if (url.endsWith("/api/lists/list-1") && method === "DELETE") {
        return new Response(null, { status: 204 });
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/"] });

    render(withRouter(router));

    expect(screen.getByText(/loading lists/i)).toBeInTheDocument();
    const settingsLink = await screen.findByTitle("Throttled");
    expect(settingsLink).toHaveAccessibleName(/settings/i);
    expect(await screen.findByText("My Lists")).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Systems" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /new list/i }));

    expect(await screen.findByRole("heading", { name: "Untitled list" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /list actions for systems/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /delete list/i }));
    expect(await screen.findByRole("dialog", { name: /delete list/i })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "Systems" })).not.toBeInTheDocument();
    });
  });

  test("renames a list and changes its thumbnail from the home page menus", async () => {
    let currentListName = "Systems";
    let currentThumbnailSourceId: string | null = null;
    let currentThumbnailUrl: string | null = null;

    installFetchMock(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url.endsWith("/api/health") && method === "GET") {
        return Response.json({
          overall: "ok",
          dependencies: {
            backend: { status: "ok", message: "" },
          },
        });
      }

      if (url.endsWith("/api/lists") && method === "GET") {
        return Response.json([
          {
            id: "list-1",
            name: currentListName,
            created_at: "2026-03-31T19:00:00Z",
            thumbnail_source_id: currentThumbnailSourceId,
            thumbnail_url: currentThumbnailUrl,
            source_count: 2,
            updated_at: "2026-03-31T19:00:00Z",
          },
        ]);
      }

      if (url.endsWith("/api/lists/list-1/sources") && method === "GET") {
        return Response.json([
          {
            id: "source-cover",
            video_id: "BV1cover",
            platform: "bilibili",
            title: "Systems Episode",
            summary: "",
            keywords: [],
            cover_url: null,
            source_url: "https://www.bilibili.com/video/BV1cover",
            duration_seconds: 0,
            uploader: "",
            language: null,
            processed_at: "2026-03-31T20:00:00Z",
          },
          {
            id: "source-extra",
            video_id: "BV1extra",
            platform: "bilibili",
            title: "Feedback Loops",
            summary: "",
            keywords: [],
            cover_url: null,
            source_url: "https://www.bilibili.com/video/BV1extra",
            duration_seconds: 0,
            uploader: "",
            language: null,
            processed_at: "2026-03-31T20:10:00Z",
          },
        ]);
      }

      if (url.endsWith("/api/lists/list-1") && method === "PATCH") {
        const body = JSON.parse(String(init?.body));
        if (typeof body.name === "string") {
          currentListName = body.name;
        }
        if ("thumbnail_source_id" in body) {
          currentThumbnailSourceId = body.thumbnail_source_id;
          currentThumbnailUrl = body.thumbnail_source_id ? `http://testserver/covers/${body.thumbnail_source_id}` : null;
        }
        return Response.json({
          id: "list-1",
          name: currentListName,
          created_at: "2026-03-31T19:00:00Z",
          thumbnail_source_id: currentThumbnailSourceId,
          thumbnail_url: currentThumbnailUrl,
          source_count: 2,
          updated_at: "2026-03-31T19:00:00Z",
        });
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/"] });
    render(withRouter(router));

    await screen.findByRole("heading", { name: "Systems" });

    await userEvent.click(screen.getByRole("button", { name: /list actions for systems/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /rename/i }));
    expect(router.state.location.pathname).toBe("/");
    expect(screen.getByTestId("home-page-content")).toHaveAttribute("aria-hidden", "true");
    const input = await screen.findByLabelText(/list name/i);
    await userEvent.clear(input);
    await userEvent.type(input, "Distributed Systems");
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    expect(await screen.findByRole("heading", { name: "Distributed Systems" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /list actions for distributed systems/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /change thumbnail/i }));
    expect(await screen.findByRole("dialog", { name: /choose thumbnail/i })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /systems episode/i }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /choose thumbnail/i })).not.toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /list actions for distributed systems/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /change thumbnail/i }));
    await userEvent.click(await screen.findByRole("button", { name: /no cover/i }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /choose thumbnail/i })).not.toBeInTheDocument();
    });
  });

  test("supports navigation across home, list workspace, and settings routes", async () => {
    installFetchMock(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url.endsWith("/api/health") && method === "GET") {
        return Response.json({
          overall: "ok",
          dependencies: {
            backend: { status: "ok", message: "" },
            llm: { status: "ok", message: "" },
            whisper_model: { status: "ok", message: "" },
            ffmpeg: { status: "ok", message: "" },
            cuda: { status: "ok", message: "" },
            bilibili_session: { status: "ok", message: "" },
            embedding_model: { status: "ok", message: "" },
          },
        });
      }

      if (url.endsWith("/api/lists") && method === "GET") {
        return Response.json([
          {
            id: "list-1",
            name: "Systems",
            created_at: "2026-03-31T19:00:00Z",
            thumbnail_source_id: null,
            thumbnail_url: null,
            source_count: 0,
            updated_at: "2026-03-31T19:00:00Z",
          },
        ]);
      }

      if (url.endsWith("/api/lists/list-1/sources") && method === "GET") {
        return Response.json([]);
      }

      if (url.endsWith("/api/config") && method === "GET") {
        return Response.json({
          accounts: { bilibili: { cookie: "***", last_verified: "" } },
          ai: { provider: "openai", model: "gpt-4o", api_key: "***", base_url: "" },
          transcription: {
            engine: "faster-whisper",
            model_size: "large-v3",
            device: "cuda",
            language: "auto",
          },
          vision: { enabled: false, frame_sample_rate: 30, model: null },
          backend: { port: 8765, worker_concurrency: 1 },
        });
      }

      if (url.endsWith("/api/models/whisper") && method === "GET") {
        return Response.json([{ name: "large-v3", installed: true, path: "/tmp/large-v3", selected: true }]);
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        return Response.json([]);
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/"] });
    render(withRouter(router));

    expect(await screen.findByRole("heading", { name: "Systems" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /job spirit/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /open systems/i }));
    expect(await screen.findByRole("heading", { name: /sources/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /job spirit/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("link", { name: /settings/i }));
    expect(await screen.findByRole("heading", { name: /settings/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /job spirit/i })).not.toBeInTheDocument();
  });

  test("shows list loading errors inline", async () => {
    installFetchMock(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url.endsWith("/api/health") && method === "GET") {
        return Response.json({
          overall: "ok",
          dependencies: {
            backend: { status: "ok", message: "" },
          },
        });
      }

      if (url.endsWith("/api/lists") && method === "GET") {
        return Response.json({ detail: "Lists unavailable" }, { status: 503 });
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        return Response.json([]);
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/"] });
    render(withRouter(router));

    expect(await screen.findByText("error.apiError")).toBeInTheDocument();
  });
});
