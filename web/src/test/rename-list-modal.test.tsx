import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { RenameListModal } from "@/components/lists/RenameListModal";
import type { BibilabList } from "@/lib/types";

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

function renderModal(props?: Partial<React.ComponentProps<typeof RenameListModal>>) {
  return render(
    <LanguageProvider>
      <RenameListModal
        list={list}
        open={true}
        onClose={vi.fn()}
        onCommit={vi.fn().mockResolvedValue(undefined)}
        initialValue="Systems"
        {...props}
      />
    </LanguageProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("RenameListModal", () => {
  test("renders with initial value in input", () => {
    renderModal();
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByLabelText("List name")).toHaveValue("Systems");
  });

  test("cancel closes without committing", async () => {
    const onClose = vi.fn();
    const onCommit = vi.fn().mockResolvedValue(undefined);
    renderModal({ onClose, onCommit });

    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onCommit).not.toHaveBeenCalled();
  });

  test("save with changed name calls onCommit then onClose", async () => {
    const onClose = vi.fn();
    const onCommit = vi.fn().mockResolvedValue(undefined);
    renderModal({ onClose, onCommit });

    const input = screen.getByLabelText("List name");
    await userEvent.clear(input);
    await userEvent.type(input, "Renamed");
    await userEvent.click(screen.getByRole("button", { name: /save/i }));

    expect(onCommit).toHaveBeenCalledWith("Renamed");
    await vi.waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  test("save with unchanged name closes without committing", async () => {
    const onClose = vi.fn();
    const onCommit = vi.fn().mockResolvedValue(undefined);
    renderModal({ onClose, onCommit });

    await userEvent.click(screen.getByRole("button", { name: /save/i }));
    expect(onCommit).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("enter key submits the form", async () => {
    const onCommit = vi.fn().mockResolvedValue(undefined);
    renderModal({ onCommit });

    const input = screen.getByLabelText("List name");
    await userEvent.clear(input);
    await userEvent.type(input, "New Name{Enter}");
    expect(onCommit).toHaveBeenCalledWith("New Name");
  });

  test("whitespace-only input closes without committing", async () => {
    const onClose = vi.fn();
    const onCommit = vi.fn().mockResolvedValue(undefined);
    renderModal({ onClose, onCommit, initialValue: "" });

    const input = screen.getByLabelText("List name");
    await userEvent.type(input, "   ");
    await userEvent.click(screen.getByRole("button", { name: /save/i }));
    expect(onCommit).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("does not render when closed", () => {
    renderModal({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("shows thumbnail when list has one", () => {
    renderModal({ list: { ...list, thumbnail_url: "https://example.com/thumb.jpg" } });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });
});
