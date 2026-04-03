import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { routes } from "../app/routes";

function installFetchMock(handler: (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>) {
  vi.stubGlobal("fetch", vi.fn(handler));
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("home page", () => {
  test("shows the floating shell, creates an untitled list from the first tile, and deletes an existing list", async () => {
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

    vi.stubGlobal("confirm", vi.fn(() => true));
    const router = createMemoryRouter(routes, { initialEntries: ["/"] });

    render(<RouterProvider router={router} />);

    expect(screen.getByText(/loading lists/i)).toBeInTheDocument();
    const settingsLink = await screen.findByTitle("Degraded");
    expect(settingsLink).toHaveAccessibleName(/settings/i);
    expect(await screen.findByRole("heading", { name: "Systems" })).toBeInTheDocument();

    const createButtons = screen.getAllByRole("button", { name: /create new list/i });
    await userEvent.click(createButtons[0]);

    expect(await screen.findByRole("heading", { name: "Untitled list" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /delete systems/i }));

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "Systems" })).not.toBeInTheDocument();
    });
  });

  test("shows a renamed list after navigating back from the list workspace", async () => {
    let currentListName = "Systems";

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

      if (url.endsWith("/api/lists/list-1") && method === "PATCH") {
        const body = JSON.parse(String(init?.body));
        currentListName = body.name;
        return Response.json({
          id: "list-1",
          name: currentListName,
          created_at: "2026-03-31T19:00:00Z",
          thumbnail_source_id: null,
          thumbnail_url: null,
          source_count: 0,
          updated_at: "2026-03-31T19:00:00Z",
        });
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/"] });

    render(<RouterProvider router={router} />);

    await userEvent.click(await screen.findByRole("button", { name: /open systems/i }));
    expect(await screen.findByRole("heading", { name: /systems/i })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /edit list name/i }));
    const input = screen.getByLabelText(/list name/i);
    await userEvent.clear(input);
    await userEvent.type(input, "Distributed Systems");
    await userEvent.tab();

    expect(await screen.findByRole("heading", { name: /distributed systems/i })).toBeInTheDocument();

    await router.navigate("/");

    expect(await screen.findByRole("heading", { name: "Distributed Systems" })).toBeInTheDocument();
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
    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("heading", { name: "Systems" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /job spirit/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /open systems/i }));
    expect(await screen.findByRole("heading", { name: /systems/i })).toBeInTheDocument();
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
    render(<RouterProvider router={router} />);

    expect(await screen.findByText("Lists unavailable")).toBeInTheDocument();
  });
});
