import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, test, vi } from "vitest";

import { ListGrid } from "@/components/lists/ListGrid";
import type { BibilabList } from "@/lib/types";

const list: BibilabList = {
  id: "list-1",
  name: "Systems Thinking",
  created_at: "2026-03-31T19:00:00Z",
  thumbnail_source_id: null,
  thumbnail_url: null,
  source_count: 12,
  updated_at: "2026-03-31T20:00:00Z",
};

describe("ListGrid", () => {
  test("renders the create tile, navigates from cards, and forwards card menu actions", async () => {
    const onCreate = vi.fn(async () => {});
    const onDelete = vi.fn(async () => {});
    const onRename = vi.fn();
    const onChangeThumbnail = vi.fn();
    const router = createMemoryRouter(
      [
        {
          path: "/",
          element: (
            <ListGrid
              busy={false}
              lists={[list]}
              onChangeThumbnail={onChangeThumbnail}
              onCreate={onCreate}
              onDelete={onDelete}
              onRename={onRename}
            />
          ),
        },
        {
          path: "/lists/:listId",
          element: <div>List detail</div>,
        },
      ],
      { initialEntries: ["/"] },
    );

    render(<RouterProvider router={router} />);

    await userEvent.click(screen.getByRole("button", { name: /new list/i }));
    expect(onCreate).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByRole("button", { name: /list actions for systems thinking/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /rename/i }));
    expect(onRename).toHaveBeenCalledWith(list);

    await userEvent.click(screen.getByRole("button", { name: /list actions for systems thinking/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /change thumbnail/i }));
    expect(onChangeThumbnail).toHaveBeenCalledWith(list);

    await userEvent.click(screen.getByRole("button", { name: /list actions for systems thinking/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /delete list/i }));
    expect(onDelete).toHaveBeenCalledWith(list);

    await userEvent.click(screen.getByRole("button", { name: /open systems thinking/i }));
    expect(router.state.location.pathname).toBe("/lists/list-1");
  });
});
