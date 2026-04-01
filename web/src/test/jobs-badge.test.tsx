import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, test, vi } from "vitest";

import { routes } from "../app/routes";
import type { Job } from "../lib/types";

function installFetchMock(
  handler: (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>,
) {
  vi.stubGlobal("fetch", vi.fn(handler));
}

describe("jobs badge", () => {
  test("shows active job count and lets the user open the drawer and cancel a job", async () => {
    const jobs: Job[] = [
      {
        id: "job-1",
        type: "video",
        source_url: "https://www.bilibili.com/video/BV1abc",
        platform: "bilibili",
        status: "transcribing",
        progress: 48,
        error: null,
        created_at: "2026-03-31T20:00:00Z",
        updated_at: "2026-03-31T20:01:00Z",
        meta: { title: "Message Queues" },
      },
      {
        id: "job-2",
        type: "video",
        source_url: "https://www.bilibili.com/video/BV1done",
        platform: "bilibili",
        status: "done",
        progress: 100,
        error: null,
        created_at: "2026-03-31T19:00:00Z",
        updated_at: "2026-03-31T19:10:00Z",
        meta: { title: "Finished Job" },
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
        return Response.json([]);
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        return Response.json(jobs);
      }

      if (url.endsWith("/api/jobs/job-1") && method === "DELETE") {
        jobs[0] = { ...jobs[0], status: "failed", error: "Cancelled by user" };
        return new Response(null, { status: 204 });
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/"] });

    render(<RouterProvider router={router} />);

    expect(await screen.findByRole("button", { name: /1 active job/i })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /1 active job/i }));

    expect(await screen.findByRole("heading", { name: /jobs/i })).toBeInTheDocument();
    expect(screen.getByText(/message queues/i)).toBeInTheDocument();
    expect(screen.getByText(/48%/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /cancel message queues/i }));

    await waitFor(() => {
      expect(screen.getByText(/cancelled by user/i)).toBeInTheDocument();
    });
  });
});
