import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { routes } from "@/app/routes";
import type { Job } from "@/lib/types";

function installFetchMock(
  handler: (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>,
) {
  vi.stubGlobal("fetch", vi.fn(handler));
}

describe("job spirit", () => {
  test("shows rehydrated ingest work and can be dismissed after completion", async () => {
    let sources = [
      {
        id: "source-1",
        video_id: "BV1abc",
        platform: "bilibili",
        title: "Message Queues",
        summary: "",
        keywords: [],
        cover_url: null,
        source_url: "https://www.bilibili.com/video/BV1abc",
        duration_seconds: 0,
        uploader: "",
        language: null,
        processed_at: "2026-03-31T20:00:00Z",
      },
    ];

    let jobsResponse: Job[] = [
      {
        id: "job-1",
        type: "ingest",
        status: "transcribing",
        progress: 48,
        error: null,
        created_at: "2026-03-31T20:00:00Z",
        updated_at: "2026-03-31T20:01:00Z",
        meta: {
          title: "New Source",
          list_id: "list-1",
          source_url: "https://www.bilibili.com/video/BV1new",
          platform: "bilibili",
        },
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
        return Response.json(sources);
      }

      if (url.endsWith("/api/ingest/url") && method === "POST") {
        return Response.json({ queued: ["job-1"], skipped: [] });
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        return Response.json(jobsResponse);
      }

      if (url.endsWith("/api/jobs/job-1") && method === "DELETE") {
        jobsResponse = [
          {
            ...jobsResponse[0],
            status: "failed",
            progress: 48,
            error: "Cancelled by user",
          },
        ];
        return new Response(null, { status: 204 });
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(
      <JobActivityProvider>
        <LanguageProvider>
          <RouterProvider router={router} />
        </LanguageProvider>
      </JobActivityProvider>,
    );

    expect(await screen.findByRole("heading", { name: /sources/i })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /jobs/i })).toBeInTheDocument();

    await userEvent.type(screen.getByPlaceholderText(/paste a bilibili url/i), "https://www.bilibili.com/video/BV1new");
    await userEvent.click(screen.getByRole("button", { name: /add source/i }));

    const spiritButton = await screen.findByRole("button", { name: /jobs/i });
    await userEvent.click(spiritButton);

    expect(await screen.findByText(/1 running/i)).toBeInTheDocument();
    expect(await screen.findByText(/new source/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /cancel new source/i }));

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /jobs/i })).not.toBeInTheDocument();
    });
  });
});
