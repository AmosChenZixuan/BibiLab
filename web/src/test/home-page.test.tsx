import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, test, vi } from "vitest";

import { routes } from "../app/routes";

function installFetchMock(handler: (input: RequestInfo | URL, init?: RequestInit) => Response | Promise<Response>) {
  vi.stubGlobal("fetch", vi.fn(handler));
}

describe("home page", () => {
  test("shows the floating shell, creates an untitled list from the first tile, and deletes an existing list", async () => {
    const lists = [
      { id: "list-1", name: "Systems", created_at: "2026-03-31T19:00:00Z" },
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
    expect(await screen.findByText(/system healthy/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /settings/i })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Systems" })).toBeInTheDocument();

    const createButtons = screen.getAllByRole("button", { name: /create new list/i });
    await userEvent.click(createButtons[0]);

    expect(await screen.findByRole("heading", { name: "Untitled list" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /delete systems/i }));

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "Systems" })).not.toBeInTheDocument();
    });
  });
});
