import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { routes } from "@/app/routes";
import { mockFetch, renderWithProviders } from "@/test/utils";

/** Wrap RouterProvider */
function withRouter(router: ReturnType<typeof createMemoryRouter>) {
  return renderWithProviders(
    <RouterProvider router={router} />,
    { providers: [JobActivityProvider, LanguageProvider] },
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
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

    mockFetch(async (input, init) => {
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

    withRouter(router);

    expect(screen.getByText(/loading lists/i)).toBeInTheDocument();
    const settingsLink = await screen.findByTitle("Healthy");
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

    mockFetch(async (input, init) => {
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
    withRouter(router);

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
    mockFetch(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url.endsWith("/api/health") && method === "GET") {
        return Response.json({
          overall: "ok",
          dependencies: {
            backend: { status: "ok", message: "" },
            llm: { status: "ok", message: "" },
            asr_model: { status: "ok", message: "" },
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
          accounts: { bilibili: { cookie: "***", username: "", avatar_url: "" } },
          ai: { protocol: "openai", model: "gpt-4o", api_key: "***", base_url: "" },
          transcription: {
            model: "large-v3",
            device: "cuda",
            language: "auto",
          },
          backend: { port: 8765, max_concurrent_jobs: 1, cors_origins: ["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:5173"] },
          rag: { max_distance: 0.8, reranking_enabled: true, hybrid_enabled: true, debug_prompts: false },
        });
      }

      if (url.endsWith("/api/models") && method === "GET") {
        return Response.json([
          { id: "large-v3", display_name: "Faster Whisper large-v3", kind: "transcription", status: "present", required_by_config: true, path: "/tmp/large-v3", size_mb: 3000 },
        ]);
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        return Response.json([]);
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/"] });
    withRouter(router);

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
    mockFetch(async (input, init) => {
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
    withRouter(router);

    expect(await screen.findByText("Lists unavailable")).toBeInTheDocument();
  });
});
