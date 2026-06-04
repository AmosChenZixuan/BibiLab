import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { PreviewVideo } from "@/lib/types";
import { LanguageProvider } from "@/app/LanguageContext";
import { PlaylistPreviewModal } from "@/components/lists/sources/PlaylistPreviewModal";
import { renderWithProviders } from "@/test/utils";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = "";
});

let _idCounter = 0;
function makeVideo(overrides: Partial<PreviewVideo> = {}): PreviewVideo {
  _idCounter++;
  const id = `v${_idCounter}`;
  return {
    video_id: id,
    title: `Test Video ${id}`,
    cover_url: "https://example.com/cover.jpg",
    duration_seconds: 180,
    uploader: "Test Author",
    platform: "bilibili",
    source_url: `https://bilibili.com/video/${id}`,
    part_label: null,
    status: "new",
    ...overrides,
  };
}

function renderModal(videos: PreviewVideo[], submitting = false) {
  const onSubmit = vi.fn();
  const onCancel = vi.fn();
  renderWithProviders(
    <PlaylistPreviewModal
      videos={videos}
      onSubmit={onSubmit}
      onCancel={onCancel}
      submitting={submitting}
    />,
    { providers: [LanguageProvider] },
  );
  return { onSubmit, onCancel };
}

function masterCheckbox(): HTMLInputElement {
  return screen.getByTestId("master-checkbox") as HTMLInputElement;
}

function rowCheckboxes(): HTMLInputElement[] {
  return screen.getAllByTestId(/^row-checkbox-/) as HTMLInputElement[];
}

describe("PlaylistPreviewModal", () => {
  // ── 1. Default selection ───────────────────────────────────────────────────
  it("checks all new videos on mount", () => {
    const videos = [makeVideo({ video_id: "v1" }), makeVideo({ video_id: "v2" }), makeVideo({ video_id: "v3" })];
    renderModal(videos);
    const master = masterCheckbox();
    const rows = rowCheckboxes();
    expect(rows).toHaveLength(3);
    expect(master).toBeChecked();
    expect(rows.every((cb) => cb.checked)).toBe(true);
  });

  // ── 2. Tri-state master checkbox ───────────────────────────────────────────
  it("master: all checked → checked=true, indeterminate=false", () => {
    const videos = [makeVideo({ video_id: "v1" }), makeVideo({ video_id: "v2" })];
    renderModal(videos);
    const master = masterCheckbox();
    expect(master).toBeChecked();
    expect(master.indeterminate).toBe(false);
  });

  it("master: unchecking one row → checked=false, indeterminate=true", async () => {
    const user = userEvent.setup();
    const videos = [makeVideo({ video_id: "v1" }), makeVideo({ video_id: "v2" })];
    renderModal(videos);
    const rows = rowCheckboxes();
    await user.click(rows[0]); // uncheck first row
    const master = masterCheckbox();
    expect(master).not.toBeChecked();
    expect(master.indeterminate).toBe(true);
  });

  it("master: unchecking all rows → checked=false, indeterminate=false", () => {
    const videos = [makeVideo({ video_id: "v1" }), makeVideo({ video_id: "v2" })];
    renderModal(videos);
    const master = masterCheckbox();
    const rows = rowCheckboxes();
    fireEvent.click(rows[0]);
    fireEvent.click(rows[1]);
    expect(master).not.toBeChecked();
    expect(master.indeterminate).toBe(false);
  });

  it("master: click when none selected → selects all", () => {
    const videos = [makeVideo({ video_id: "v1" }), makeVideo({ video_id: "v2" })];
    renderModal(videos);
    const master = masterCheckbox();
    // uncheck all via rows first
    const rows = rowCheckboxes();
    fireEvent.click(rows[0]);
    fireEvent.click(rows[1]);
    expect(master).not.toBeChecked();
    // now click master to select all
    fireEvent.click(master);
    const rowsAfter = rowCheckboxes();
    expect(rowsAfter.every((cb) => cb.checked)).toBe(true);
  });

  it("master: click when all selected → clears all", () => {
    const videos = [makeVideo({ video_id: "v1" }), makeVideo({ video_id: "v2" })];
    renderModal(videos);
    const master = masterCheckbox();
    expect(master).toBeChecked();
    fireEvent.click(master);
    expect(master).not.toBeChecked();
  });

  // ── 3. Already-in-list section ─────────────────────────────────────────────
  it("hides Already-in-list section when all videos are new", () => {
    const videos = [makeVideo({ status: "new" }), makeVideo({ status: "new" })];
    renderModal(videos);
    expect(screen.queryByText("Already in list")).not.toBeInTheDocument();
  });

  it("shows Already-in-list section when mixed statuses", () => {
    const videos = [
      makeVideo({ video_id: "v1", status: "new" }),
      makeVideo({ video_id: "v2", status: "processed" }),
    ];
    renderModal(videos);
    expect(screen.getByText("Already in list")).toBeInTheDocument();
  });

  // ── 4. Status badges ───────────────────────────────────────────────────────
  it("renders processed badge for processed videos", () => {
    const videos = [makeVideo({ video_id: "v1", status: "processed" })];
    renderModal(videos);
    expect(screen.getByText("Processed")).toBeInTheDocument();
  });

  it("renders in_progress badge for in_progress videos", () => {
    const videos = [makeVideo({ video_id: "v1", status: "in_progress" })];
    renderModal(videos);
    expect(screen.getByText("In progress")).toBeInTheDocument();
  });

  it("renders needs_auth badge for needs_auth videos", () => {
    const videos = [makeVideo({ video_id: "v1", status: "needs_auth" })];
    renderModal(videos);
    expect(screen.getByText("Needs auth")).toBeInTheDocument();
  });

  // ── 5. part_label badge ────────────────────────────────────────────────────
  it("shows part_label badge when part_label is not null", () => {
    const videos = [makeVideo({ video_id: "v1", part_label: "P3" })];
    renderModal(videos);
    expect(screen.getByText("P3")).toBeInTheDocument();
  });

  it("hides part_label badge when part_label is null", () => {
    const videos = [makeVideo({ video_id: "v1", part_label: null })];
    renderModal(videos);
    expect(screen.queryByText("P3")).not.toBeInTheDocument();
  });

  // ── 6. Submit payload shape ────────────────────────────────────────────────
  it("onSubmit called with IngestVideoIn shape (no status, no part_label)", async () => {
    const user = userEvent.setup();
    const videos = [
      makeVideo({ video_id: "v1", title: "Video 1", status: "new", part_label: "P1" }),
      makeVideo({ video_id: "v2", title: "Video 2", status: "new", part_label: null }),
      makeVideo({ video_id: "v3", title: "Video 3", status: "new", part_label: null }),
    ];
    const { onSubmit } = renderModal(videos);
    const rows = rowCheckboxes();
    await user.click(rows[1]); // uncheck second video
    const submitBtn = screen.getByRole("button", { name: "Add selected" });
    await user.click(submitBtn);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const [[payload]] = onSubmit.mock.calls;
    expect(payload).toHaveLength(2);
    for (const video of payload) {
      expect("status" in video).toBe(false);
      expect("part_label" in video).toBe(false);
    }
  });

  // ── 7. Submit disabled ────────────────────────────────────────────────────
  it("submit disabled when zero selected", async () => {
    const user = userEvent.setup();
    const videos = [makeVideo({ video_id: "v1" })];
    renderModal(videos);
    const master = masterCheckbox();
    fireEvent.click(master); // uncheck all
    const submitBtn = screen.getByRole("button", { name: "Add selected" });
    expect(submitBtn).toBeDisabled();
  });

  it("submit disabled when submitting=true", () => {
    const videos = [makeVideo({ video_id: "v1" })];
    renderModal(videos, true);
    const submitBtn = screen.getByRole("button", { name: /add selected/i });
    expect(submitBtn).toBeDisabled();
  });

  // ── 8. Cancel ─────────────────────────────────────────────────────────────
  it("cancel calls onCancel exactly once", async () => {
    const user = userEvent.setup();
    const videos = [makeVideo({ video_id: "v1" })];
    const { onCancel, onSubmit } = renderModal(videos);
    const cancelBtn = screen.getByRole("button", { name: "Cancel" });
    await user.click(cancelBtn);
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
