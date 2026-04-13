import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PreviewResponse, PreviewVideo } from "@/lib/types";
import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { SourcesListMode } from "@/components/lists/sources/SourcesListMode";

function makeVideo(overrides: Partial<PreviewVideo> = {}): PreviewVideo {
  return {
    video_id: "BVtest",
    title: "Test Video",
    cover_url: "https://example.com/cover.jpg",
    duration_seconds: 180,
    uploader: "Test Author",
    platform: "bilibili",
    source_url: "https://bilibili.com/video/BVtest",
    part_label: null,
    status: "new",
    ...overrides,
  };
}

const state = {
  previewResponse: null as PreviewResponse | null,
  ingestResult: { queued: [] as string[], skipped: [] as string[] },
  ingestCalls: [] as Array<{ listId: string; videos: unknown[] }>,
};

vi.mock("@/lib/api", () => {
  const mockPreviewPlaylist = vi.fn((listId: string, url: string) => {
    return Promise.resolve(state.previewResponse ?? { videos: [] });
  });
  const mockIngestUrl = vi.fn((listId: string, videos: unknown[]) => {
    state.ingestCalls.push({ listId, videos });
    return Promise.resolve(state.ingestResult);
  });
  const mockTrackJobs = vi.fn();
  const mockListSources = vi.fn().mockResolvedValue([]);
  const mockDismissJob = vi.fn().mockResolvedValue(undefined);
  const mockGetJobs = vi.fn().mockReturnValue([]);
  return {
    api: {
      previewPlaylist: mockPreviewPlaylist,
      ingestUrl: mockIngestUrl,
      listSources: mockListSources,
      deleteSource: vi.fn().mockResolvedValue(undefined),
    },
    createApiClient: () => ({
      previewPlaylist: mockPreviewPlaylist,
      ingestUrl: mockIngestUrl,
      listSources: mockListSources,
      deleteSource: vi.fn().mockResolvedValue(undefined),
    }),
    toErrorMessageWithT: vi.fn((e: unknown) => (e instanceof Error ? e.message : String(e))),
    toErrorMessage: vi.fn((e: unknown) => (e instanceof Error ? e.message : String(e))),
    default: {
      previewPlaylist: mockPreviewPlaylist,
      ingestUrl: mockIngestUrl,
      listSources: mockListSources,
      deleteSource: vi.fn().mockResolvedValue(undefined),
    },
  };
});

import { api } from "@/lib/api";

function renderMode(sources: Parameters<typeof SourcesListMode>[0]["sources"] = []) {
  const trackJobs = vi.fn();
  const getJobs = vi.fn().mockReturnValue([]);
  const dismissJob = vi.fn().mockResolvedValue(undefined);

  const result = render(
    <LanguageProvider>
      <JobActivityProvider>
        <SourcesListMode
          listId="list-1"
          sources={sources}
          onOpenSource={vi.fn()}
        />
      </JobActivityProvider>
    </LanguageProvider>,
  );
  return { ...result, trackJobs, getJobs, dismissJob };
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  state.previewResponse = null;
  state.ingestResult = { queued: ["job-1"], skipped: [] };
  state.ingestCalls = [];
});

// ─── Tests ─────────────────────────────────────────────────────────────────────

describe("SourcesListMode preview flow", () => {
  it("shows error when single video is already processed", async () => {
    state.previewResponse = {
      videos: [makeVideo({ video_id: "BV1", status: "processed" })],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    await userEvent.type(input, "https://bilibili.com/video/BV1");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Already in this list", { exact: false })).toBeInTheDocument();
    });
  });

  it("auto-submits single new video without opening modal", async () => {
    state.previewResponse = {
      videos: [makeVideo({ video_id: "BV1", title: "New Video" })],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    await userEvent.type(input, "https://bilibili.com/video/BV1");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(api.ingestUrl).toHaveBeenCalledWith(
        "list-1",
        expect.arrayContaining([
          expect.objectContaining({ video_id: "BV1", title: "New Video" }),
        ]),
      );
    });
  });

  it("opens modal when multiple new videos returned", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", title: "Video 1" }),
        makeVideo({ video_id: "BV2", title: "Video 2" }),
      ],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /preview playlist/i })).toBeInTheDocument();
    });
  });

  it("shows error when all videos are already processed", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", status: "processed" }),
        makeVideo({ video_id: "BV2", status: "processed" }),
      ],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText(/all 2 videos are already in this list/i)).toBeInTheDocument();
    });
  });

  it("shows needs_auth error when all videos need auth", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", status: "needs_auth" }),
        makeVideo({ video_id: "BV2", status: "needs_auth" }),
      ],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText(/need authentication/i)).toBeInTheDocument();
    });
  });

  it("modal submit calls ingestUrl with selected videos", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", title: "Video 1" }),
        makeVideo({ video_id: "BV2", title: "Video 2" }),
      ],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /preview playlist/i })).toBeInTheDocument();
    });

    const submitBtn = screen.getByRole("button", { name: /add selected/i });
    await userEvent.click(submitBtn);

    await waitFor(() => {
      expect(api.ingestUrl).toHaveBeenCalledWith(
        "list-1",
        expect.arrayContaining([
          expect.objectContaining({ video_id: "BV1" }),
          expect.objectContaining({ video_id: "BV2" }),
        ]),
      );
    });
  });

  it("modal cancel closes modal without submitting", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", title: "Video 1" }),
        makeVideo({ video_id: "BV2", title: "Video 2" }),
      ],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /preview playlist/i })).toBeInTheDocument();
    });

    const cancelBtn = screen.getByRole("button", { name: /cancel/i });
    await userEvent.click(cancelBtn);

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: /preview playlist/i })).not.toBeInTheDocument();
    });
    expect(api.ingestUrl).not.toHaveBeenCalled();
  });

  it("shows partial skip error when some videos were already in list", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", title: "Video 1" }),
        makeVideo({ video_id: "BV2", title: "Video 2" }),
      ],
    };
    state.ingestResult = { queued: ["job-1"], skipped: ["BV2"] };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a bilibili url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /preview playlist/i })).toBeInTheDocument();
    });

    const submitBtn = screen.getByRole("button", { name: /add selected/i });
    await userEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/1 of 2 queued/i)).toBeInTheDocument();
    });
  });
});
