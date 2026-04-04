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

  test("shows list name in navbar and commits rename on blur", async () => {
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

      if (url.endsWith("/api/lists/list-1") && method === "PATCH") {
        const body = JSON.parse(String(init?.body));
        expect(body).toEqual({ name: "New Name" });
        return Response.json({
          id: "list-1",
          name: "New Name",
          created_at: "2026-03-31T19:00:00Z",
          thumbnail_source_id: null,
          thumbnail_url: null,
          source_count: 0,
          updated_at: "2026-03-31T21:00:00Z",
        });
      }

      throw new Error(`Unhandled ${method} ${url}`);
    });

    const router = createMemoryRouter(routes, { initialEntries: ["/lists/list-1"] });
    render(<RouterProvider router={router} />);

    // List name is visible in navbar (portaled into <nav>)
    const nav = document.querySelector("nav");
    expect(nav).toBeInTheDocument();
    expect(await screen.findByText("AI Reading List")).toBeInTheDocument();

    // Clicking the name enters edit mode (per spec: click name to edit inline)
    await userEvent.click(screen.getByText("AI Reading List"));

    // After clicking, an input is visible
    const input = screen.getByRole("textbox");
    expect(input).toBeInTheDocument();
    expect(input).toHaveValue("AI Reading List");

    // Type a new name and blur → commits PATCH
    await userEvent.clear(input);
    await userEvent.type(input, "New Name");
    await userEvent.tab(); // blur

    // After PATCH resolves, the updated name is shown
    expect(await screen.findByText("New Name")).toBeInTheDocument();

    // Escape reverts without firing PATCH
    await userEvent.click(screen.getByText("New Name"));
    const input2 = screen.getByRole("textbox");
    await userEvent.clear(input2);
    await userEvent.type(input2, "Should Not Commit");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByText("Should Not Commit")).not.toBeInTheDocument();
    expect(screen.getByText("New Name")).toBeInTheDocument();
  });
});
