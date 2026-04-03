import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { routes } from "../app/routes";

const downloadTextFile = vi.fn();

vi.mock("../lib/download", () => ({
  downloadTextFile: (...args: unknown[]) => downloadTextFile(...args),
}));

afterEach(() => {
  cleanup();
  downloadTextFile.mockReset();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function installFetchMock(
  handler: (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>,
) {
  vi.stubGlobal("fetch", vi.fn(handler));
}

describe("list detail page", () => {
  test("renders the three-panel workspace, supports inline list rename on blur, source detail tabs, source deletion, and overview download", async () => {
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
            name: "Systems",
            created_at: "2026-03-31T19:00:00Z",
            thumbnail_source_id: null,
            thumbnail_url: null,
            source_count: 1,
            updated_at: "2026-03-31T20:00:00Z",
          },
        ]);
      }

      if (url.endsWith("/api/lists/list-1/sources") && method === "GET") {
        return Response.json([
          {
            video_id: "BV1abc",
            platform: "bilibili",
            title: "Message Queues",
            note_path: "/tmp/BV1abc.md",
            processed_at: "2026-03-31T20:00:00Z",
          },
        ]);
      }

      if (url.endsWith("/api/lists/list-1") && method === "PATCH") {
        const body = JSON.parse(String(init?.body));
        expect(body).toEqual({ name: "Distributed Systems" });
        return Response.json({
          id: "list-1",
          name: "Distributed Systems",
          created_at: "2026-03-31T19:00:00Z",
          thumbnail_source_id: null,
          thumbnail_url: null,
          source_count: 1,
          updated_at: "2026-03-31T20:00:00Z",
        });
      }

      if (url.endsWith("/api/ingest/url") && method === "POST") {
        const body = JSON.parse(String(init?.body));
        expect(body).toEqual({ list_id: "list-1", url: "https://www.bilibili.com/video/BV1new" });
        return Response.json({ queued: ["job-1"], skipped: [] });
      }

      if (url.endsWith("/api/notes/BV1abc/content") && method === "GET") {
        return Response.json({
          video_id: "BV1abc",
          title: "Message Queues",
          markdown: "# Message Queues\n\n## Summary\nQueues smooth bursty traffic.",
        });
      }

      if (url.endsWith("/api/notes/BV1abc/transcript") && method === "GET") {
        return Response.json({
          video_id: "BV1abc",
          text: "[00:00:02] Queues absorb spikes.",
        });
      }

      if (url.endsWith("/api/lists/list-1/overview") && method === "POST") {
        return Response.json({
          filename: "overview-systems.md",
          content: "# Systems - Overview",
        });
      }

      if (url.endsWith("/api/lists/list-1/sources/BV1abc") && method === "DELETE") {
        return new Response(null, { status: 204 });
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    vi.stubGlobal("confirm", vi.fn(() => true));
    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });

    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("heading", { name: /systems/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /sources/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /studio/i })).toBeInTheDocument();
    expect(screen.getByText(/chat arrives in v1/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /edit list name/i }));
    const nameInput = screen.getByLabelText(/list name/i);
    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, "Distributed Systems");
    await userEvent.tab();
    expect(await screen.findByRole("heading", { name: /distributed systems/i })).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/source url/i), "https://www.bilibili.com/video/BV1new");
    await userEvent.click(screen.getByRole("button", { name: /queue source/i }));
    expect(await screen.findByText(/queued 1 source/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /open message queues/i }));
    expect(await screen.findAllByText(/message queues/i)).toHaveLength(2);
    expect(await screen.findByText(/queues smooth bursty traffic/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /transcript/i }));
    expect(await screen.findByText(/\[00:00:02\] queues absorb spikes./i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /back to sources/i }));
    await userEvent.click(screen.getByRole("button", { name: /delete message queues/i }));
    await waitFor(() => {
      expect(screen.queryByText("Message Queues")).not.toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /generate overview/i }));
    await waitFor(() => {
      expect(downloadTextFile).toHaveBeenCalledWith("overview-systems.md", "# Systems - Overview");
    });
  });

  test("supports ingest reruns and reports queued and skipped source counts", async () => {
    const sources = [
      {
        video_id: "BV1old",
        platform: "bilibili",
        title: "Existing Source",
        note_path: "/tmp/BV1old.md",
        processed_at: "2026-03-31T20:00:00Z",
      },
    ];
    let jobPollCount = 0;

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
            name: "Systems",
            created_at: "2026-03-31T19:00:00Z",
            thumbnail_source_id: null,
            thumbnail_url: null,
            source_count: 1,
            updated_at: "2026-03-31T20:00:00Z",
          },
        ]);
      }

      if (url.endsWith("/api/lists/list-1/sources") && method === "GET") {
        return Response.json(sources);
      }

      if (url.endsWith("/api/ingest/url") && method === "POST") {
        const body = JSON.parse(String(init?.body));
        expect(body).toEqual({ list_id: "list-1", url: "https://www.bilibili.com/video/BV1playlist" });
        sources.push({
          video_id: "BV1new-1",
          platform: "bilibili",
          title: "Partitioning Part 1",
          note_path: "/tmp/BV1new-1.md",
          processed_at: "2026-03-31T21:00:00Z",
        });
        sources.push({
          video_id: "BV1new-2",
          platform: "bilibili",
          title: "Partitioning Part 2",
          note_path: "/tmp/BV1new-2.md",
          processed_at: "2026-03-31T21:05:00Z",
        });
        return Response.json({ queued: ["job-1", "job-2"], skipped: ["BV1old"] });
      }

      if (url.endsWith("/api/ingest/url?rerun=true") && method === "POST") {
        const body = JSON.parse(String(init?.body));
        expect(body).toEqual({ list_id: "list-1", url: "https://www.bilibili.com/video/BV1old" });
        return Response.json({ queued: ["job-3"], skipped: [] });
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        jobPollCount += 1;
        if (jobPollCount === 1) {
          return Response.json([
            {
              id: "job-1",
              type: "ingest",
              status: "done",
              progress: 100,
              error: null,
              created_at: "2026-03-31T21:00:00Z",
              updated_at: "2026-03-31T21:10:00Z",
              meta: {
                title: "Partitioning Part 1",
                list_id: "list-1",
                source_url: "https://www.bilibili.com/video/BV1playlist",
                platform: "bilibili",
              },
            },
            {
              id: "job-2",
              type: "ingest",
              status: "done",
              progress: 100,
              error: null,
              created_at: "2026-03-31T21:01:00Z",
              updated_at: "2026-03-31T21:11:00Z",
              meta: {
                title: "Partitioning Part 2",
                list_id: "list-1",
                source_url: "https://www.bilibili.com/video/BV1playlist",
                platform: "bilibili",
              },
            },
          ]);
        }

        return Response.json([
          {
            id: "job-3",
            type: "ingest",
            status: "done",
            progress: 100,
            error: null,
            created_at: "2026-03-31T21:12:00Z",
            updated_at: "2026-03-31T21:13:00Z",
            meta: {
              title: "Existing Source",
              list_id: "list-1",
              source_url: "https://www.bilibili.com/video/BV1old",
              platform: "bilibili",
            },
          },
        ]);
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("heading", { name: /systems/i })).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/source url/i), "https://www.bilibili.com/video/BV1playlist");
    await userEvent.click(screen.getByRole("button", { name: /queue source/i }));

    expect(await screen.findByText(/queued 2 sources and skipped 1 source/i)).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /open partitioning part 1/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /open partitioning part 2/i })).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/source url/i), "https://www.bilibili.com/video/BV1old");
    await userEvent.click(screen.getByLabelText(/re-run existing source/i));
    await userEvent.click(screen.getByRole("button", { name: /queue source/i }));

    expect(await screen.findByText(/queued 1 source/i)).toBeInTheDocument();
  });

  test("surfaces auth-required ingest failures with a readable message", async () => {
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

      if (url.endsWith("/api/ingest/url") && method === "POST") {
        return Response.json(
          {
            detail: { message: "Authentication required", resource_type: "course" },
          },
          { status: 401 },
        );
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        return Response.json([]);
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("heading", { name: /systems/i })).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/source url/i), "https://www.bilibili.com/video/BV1private");
    await userEvent.click(screen.getByRole("button", { name: /queue source/i }));

    expect(await screen.findByText("Authentication required")).toBeInTheDocument();
  });

  test("shows route-level source loading errors inline", async () => {
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
        return Response.json({ detail: "List not found" }, { status: 404 });
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        return Response.json([]);
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    expect(await screen.findByText("List not found")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /studio/i })).toBeInTheDocument();
  });
});
