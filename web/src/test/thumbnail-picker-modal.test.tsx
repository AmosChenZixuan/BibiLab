import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { ThumbnailPickerModal } from "@/components/lists/ThumbnailPickerModal";
import { api } from "@/lib/api";
import type { BibilabList, Source } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  api: {
    listSources: vi.fn().mockResolvedValue([]),
  },
  setCurrentLang: vi.fn(),
}));

const list: BibilabList = {
  id: "list-1",
  name: "Systems",
  created_at: "2026-01-01T00:00:00Z",
  thumbnail_source_id: null,
  thumbnail_url: null,
  source_count: 2,
  updated_at: "2026-01-01T00:00:00Z",
};

const sources: Source[] = [
  {
    id: "src-1",
    video_id: "BV1abc",
    platform: "bilibili",
    title: "Intro to ML",
    summary: "A great intro.",
    keywords: ["ml"],
    cover_url: "https://example.com/cover1.jpg",
    source_url: "https://bilibili.com/video/BV1abc",
    duration_seconds: 600,
    uploader: "Teacher",
    language: "en",
    processed_at: "2026-01-01T00:00:00Z",
  },
  {
    id: "src-2",
    video_id: "BV2def",
    platform: "bilibili",
    title: "Deep Learning",
    summary: "Advanced concepts.",
    keywords: ["dl"],
    cover_url: null,
    source_url: "https://bilibili.com/video/BV2def",
    duration_seconds: 1200,
    uploader: "Teacher",
    language: "en",
    processed_at: "2026-01-02T00:00:00Z",
  },
];

function renderModal(props?: Partial<React.ComponentProps<typeof ThumbnailPickerModal>>) {
  return render(
    <LanguageProvider>
      <ThumbnailPickerModal
        list={list}
        open={true}
        onClose={vi.fn()}
        onSelect={vi.fn().mockResolvedValue(undefined)}
        {...props}
      />
    </LanguageProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ThumbnailPickerModal", () => {
  test("fetches sources on open and renders thumbnails", async () => {
    vi.mocked(api.listSources).mockResolvedValue(sources);
    renderModal();

    await waitFor(() => {
      expect(screen.getByText("Intro to ML")).toBeInTheDocument();
      expect(screen.getByText("Deep Learning")).toBeInTheDocument();
    });
    expect(api.listSources).toHaveBeenCalledWith(list.id, expect.any(Object));
  });

  test("shows no-cover option that selects null", async () => {
    vi.mocked(api.listSources).mockResolvedValue(sources);
    const onSelect = vi.fn().mockResolvedValue(undefined);
    renderModal({ onSelect });

    await waitFor(() => expect(screen.getByText("Intro to ML")).toBeInTheDocument());

    const buttons = screen.getAllByRole("button");
    const noCoverButton = buttons.find((btn) => btn.textContent?.match(/no cover/i));
    expect(noCoverButton).toBeTruthy();
    await userEvent.click(noCoverButton!);
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  test("clicking a source thumbnail calls onSelect with source id", async () => {
    vi.mocked(api.listSources).mockResolvedValue(sources);
    const onSelect = vi.fn().mockResolvedValue(undefined);
    renderModal({ onSelect });

    await waitFor(() => expect(screen.getByText("Intro to ML")).toBeInTheDocument());
    await userEvent.click(screen.getByText("Intro to ML"));
    expect(onSelect).toHaveBeenCalledWith("src-1");
  });

  test("does not fetch when closed", () => {
    renderModal({ open: false });
    expect(api.listSources).not.toHaveBeenCalled();
  });

  test("handles API returning undefined", async () => {
    vi.mocked(api.listSources).mockResolvedValue(undefined as unknown as Source[]);
    renderModal();

    await waitFor(() => {
      expect(screen.queryByText("Intro to ML")).not.toBeInTheDocument();
    });
  });
});
