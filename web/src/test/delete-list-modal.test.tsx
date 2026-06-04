import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { DeleteListModal } from "@/components/lists/DeleteListModal";
import type { BibilabList } from "@/lib/types";
import { renderWithProviders } from "@/test/utils";

vi.mock("@/lib/api", () => ({ api: {}, setCurrentLang: vi.fn() }));

const list: BibilabList = {
  id: "list-1",
  name: "Systems",
  created_at: "2026-01-01T00:00:00Z",
  thumbnail_source_id: null,
  thumbnail_url: null,
  source_count: 3,
  updated_at: "2026-01-01T00:00:00Z",
};

function renderModal(props?: Partial<React.ComponentProps<typeof DeleteListModal>>) {
  return renderWithProviders(
    <DeleteListModal
      list={list}
      open={true}
      onClose={vi.fn()}
      onConfirm={vi.fn().mockResolvedValue(undefined)}
      {...props}
    />,
    { providers: [LanguageProvider] },
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("DeleteListModal", () => {
  test("renders warning with list name and source count", () => {
    renderModal();
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText(/Systems/)).toBeInTheDocument();
  });

  test("cancel button calls onClose without confirming", async () => {
    const onClose = vi.fn();
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    renderModal({ onClose, onConfirm });

    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  test("delete button calls onConfirm with the list then onClose", async () => {
    const onClose = vi.fn();
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    renderModal({ onClose, onConfirm });

    await userEvent.click(screen.getByRole("button", { name: /delete/i }));
    expect(onConfirm).toHaveBeenCalledWith(list);
    await vi.waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  test("does not render when closed", () => {
    renderModal({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("handles null list gracefully", async () => {
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    renderModal({ list: null, onConfirm });

    await userEvent.click(screen.getByRole("button", { name: /delete/i }));
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
