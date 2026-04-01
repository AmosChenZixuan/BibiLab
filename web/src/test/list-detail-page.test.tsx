import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, test, vi } from "vitest";

import { routes } from "../app/routes";

const downloadTextFile = vi.fn();

vi.mock("../lib/download", () => ({
  downloadTextFile: (...args: unknown[]) => downloadTextFile(...args),
}));

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
          { id: "list-1", name: "Systems", created_at: "2026-03-31T19:00:00Z" },
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
});
