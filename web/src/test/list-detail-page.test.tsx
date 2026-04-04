import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";

import { routes } from "../app/routes";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function installFetchMock(
  handler: (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>,
) {
  vi.stubGlobal("fetch", vi.fn(handler));
}

describe("list detail page", () => {
  test("renders three-panel workspace with correct panel headings and skeleton text", async () => {
    installFetchMock(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url.endsWith("/api/lists") && method === "GET") {
        return Response.json([
          {
            id: "list-1",
            name: "AI Reading List",
            created_at: "2026-03-31T19:00:00Z",
            thumbnail_source_id: null,
            thumbnail_url: null,
            source_count: 0,
            updated_at: "2026-03-31T20:00:00Z",
          },
        ]);
      }

      if (url.endsWith("/api/lists/list-1/sources") && method === "GET") {
        return Response.json([]);
      }

      if (url.endsWith("/api/jobs") && method === "GET") {
        return Response.json([]);
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    // Three panel headings
    expect(screen.getByRole("heading", { name: /sources/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /lab/i })).toBeInTheDocument();

    // Skeleton panel notes
    expect(screen.getByText(/list-scoped chat arrives in v1/i)).toBeInTheDocument();
    expect(screen.getByText(/synthesis tools/i)).toBeInTheDocument();

    // Collapse button
    const collapseBtn = screen.getByRole("button", { name: /collapse sources/i });
    expect(collapseBtn).toBeInTheDocument();

    // Clicking collapse changes label to "Expand sources"
    await userEvent.click(collapseBtn);
    expect(screen.getByRole("button", { name: /expand sources/i })).toBeInTheDocument();

    // Clicking expand restores
    await userEvent.click(screen.getByRole("button", { name: /expand sources/i }));
    expect(screen.getByRole("button", { name: /collapse sources/i })).toBeInTheDocument();
  });
});
