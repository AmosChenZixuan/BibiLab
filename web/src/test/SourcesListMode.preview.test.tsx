import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  IngestJob,
  IngestVideoIn,
  Job,
  PreviewResponse,
  PreviewVideo,
  Source,
  VideoMetadataMap,
} from "@/lib/types";
import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { SourcesListMode } from "@/components/lists/sources/SourcesListMode";
import { renderWithProviders } from "@/test/utils";

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

function makeSource(overrides: Partial<Source> = {}): Source {
  return {
    id: "src-1",
    video_id: "BVtest",
    platform: "bilibili",
    title: "Test Source",
    cover_url: "https://example.com/cover.jpg",
    source_url: "https://bilibili.com/video/BVtest",
    duration_seconds: 180,
    uploader: "Test Author",
    language: null,
    processed_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

const state = {
  previewResponse: null as PreviewResponse | null,
  metadataResponse: null as VideoMetadataMap | null,
  ingestResult: { queued: [] as string[], skipped: [] as string[] },
  ingestCalls: [] as Array<{ listId: string; videos: IngestVideoIn[] }>,
  jobs: [] as Job[],
};

function makeFailedIngestJob(overrides: Partial<IngestJob> = {}): IngestJob {
  return {
    id: "job-failed-1",
    type: "ingest",
    status: "failed",
    progress: 0,
    error: "[transcribing] audio_truncated",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    meta: {
      list_id: "list-1",
      video_id: "BV1",
      title: "Failed Video",
      source_url: "https://bilibili.com/video/BV1",
      platform: "bilibili",
    },
    ...overrides,
  };
}

vi.mock("@/lib/api", async () => {
  const { createMockApi } = await import("@/test/utils");
  const mockPreviewPlaylist = vi.fn((listId: string, url: string) => {
    return Promise.resolve(state.previewResponse ?? { videos: [] });
  });
  const mockPreviewPlaylistMetadata = vi.fn((videoIds: string[], platform: string) => {
    return Promise.resolve(
      state.metadataResponse ?? { videos: {}, expanded: {} },
    );
  });
  const mockIngestUrl = vi.fn((listId: string, videos: IngestVideoIn[]) => {
    state.ingestCalls.push({ listId, videos });
    return Promise.resolve(state.ingestResult);
  });
  const mockListSources = vi.fn().mockResolvedValue([]);
  const mockApi = createMockApi({
    previewPlaylist: mockPreviewPlaylist,
    previewPlaylistMetadata: mockPreviewPlaylistMetadata,
    ingestUrl: mockIngestUrl,
    listSources: mockListSources,
    listJobs: vi.fn(() => Promise.resolve(state.jobs)),
    deleteJob: vi.fn().mockResolvedValue(undefined),
    deleteSource: vi.fn().mockResolvedValue(undefined),
    auth: {
      generateBilibiliQr: vi.fn().mockResolvedValue(null),
      pollBilibiliQr: vi.fn().mockResolvedValue(null),
      deleteBilibiliAuth: vi.fn().mockResolvedValue(undefined),
    },
  });
  class MockApiError extends Error {
    status: number;
    detail: unknown;
    constructor(status: number, detail: unknown) {
      super(typeof detail === "string" ? detail : "Request failed");
      this.name = "ApiError";
      this.status = status;
      this.detail = detail;
    }
  }
  return {
    api: mockApi,
    ApiError: MockApiError,
    createApiClient: () => mockApi,
    toErrorMessageWithT: vi.fn((e: unknown) => (e instanceof Error ? e.message : String(e))),
    default: mockApi,
  };
});

import { api, ApiError } from "@/lib/api";

function renderMode(
  sources: Parameters<typeof SourcesListMode>[0]["sources"] = [],
  selectedSourceIds: string[] = [],
  onSelectedSourcesChange: (ids: string[]) => void = vi.fn(),
  onRefresh: () => void = vi.fn(),
) {
  return renderWithProviders(
    <SourcesListMode
      listId="list-1"
      sources={sources}
      selectedSourceIds={selectedSourceIds}
      onSelectedSourcesChange={onSelectedSourcesChange}
      onRefresh={onRefresh}
      onOpenSource={vi.fn()}
    />,
    { providers: [LanguageProvider, JobActivityProvider] },
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  state.previewResponse = null;
  state.metadataResponse = null;
  state.ingestResult = { queued: ["job-1"], skipped: [] };
  state.ingestCalls = [];
  state.jobs = [];
});

// ─── Tests ─────────────────────────────────────────────────────────────────────

describe("SourcesListMode preview flow", () => {
  it("shows error when single video is already processed", async () => {
    state.previewResponse = {
      videos: [makeVideo({ video_id: "BV1", status: "processed" })],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a video url/i);
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

    const input = screen.getByPlaceholderText(/paste a video url/i);
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

    const input = screen.getByPlaceholderText(/paste a video url/i);
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

    const input = screen.getByPlaceholderText(/paste a video url/i);
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

    const input = screen.getByPlaceholderText(/paste a video url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText(/need authentication/i)).toBeInTheDocument();
    });
  });

  it("opens QR modal when preview returns 401", async () => {
    vi.mocked(api.previewPlaylist).mockRejectedValueOnce(
      new ApiError(401, "Unauthorized")
    );
    renderMode();

    const input = screen.getByPlaceholderText(/paste a video url/i);
    await userEvent.type(input, "https://space.bilibili.com/123/favlist?fid=456");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Sign in to Bilibili")).toBeInTheDocument();
    });
    expect(screen.queryByText(/failed/i)).not.toBeInTheDocument();
  });

  it("shows auth-required error instead of QR modal when a non-bilibili preview returns 401", async () => {
    vi.mocked(api.previewPlaylist).mockRejectedValueOnce(
      new ApiError(401, "Unauthorized")
    );
    renderMode();

    const input = screen.getByPlaceholderText(/paste a video url/i);
    await userEvent.type(input, "https://www.youtube.com/watch?v=abc123xyz00");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText(/requires signing in/i)).toBeInTheDocument();
    });
    expect(screen.queryByText("Sign in to Bilibili")).not.toBeInTheDocument();
  });

  it("modal submit calls ingestUrl with selected videos", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", title: "Video 1" }),
        makeVideo({ video_id: "BV2", title: "Video 2" }),
      ],
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a video url/i);
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

    const input = screen.getByPlaceholderText(/paste a video url/i);
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

    const input = screen.getByPlaceholderText(/paste a video url/i);
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

  it("test_preview_loads_metadata_after_flat", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", title: "Video 1", cover_url: "", duration_seconds: 0, uploader: "" }),
        makeVideo({ video_id: "BV2", title: "Video 2", cover_url: "", duration_seconds: 0, uploader: "" }),
      ],
    };
    state.metadataResponse = {
      videos: {
        BV1: { title: "Video 1", cover_url: "https://example.com/c1.jpg", duration_seconds: 120, uploader: "Author1", source_url: "https://bilibili.com/video/BV1", part_label: null },
        BV2: { title: "Video 2", cover_url: "https://example.com/c2.jpg", duration_seconds: 240, uploader: "Author2", source_url: "https://bilibili.com/video/BV2", part_label: null },
      },
      expanded: {},
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a video url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /preview playlist/i })).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(api.previewPlaylist).toHaveBeenCalledWith("list-1", "https://bilibili.com/playlist/BVlist");
    });

    await waitFor(() => {
      expect(api.previewPlaylistMetadata).toHaveBeenCalledWith(["BV1", "BV2"], "bilibili");
    });
  });

  it("metadata endpoint called with array of video IDs", async () => {
    state.previewResponse = {
      videos: [
        makeVideo({ video_id: "BV1", title: "V1" }),
        makeVideo({ video_id: "BV2", title: "V2" }),
      ],
    };
    state.metadataResponse = {
      videos: {
        BV1: { title: "V1", cover_url: "c.jpg", duration_seconds: 100, uploader: "U", source_url: "", part_label: null },
        BV2: { title: "V2", cover_url: "c.jpg", duration_seconds: 200, uploader: "U", source_url: "", part_label: null },
      },
      expanded: {},
    };
    renderMode();

    const input = screen.getByPlaceholderText(/paste a video url/i);
    await userEvent.type(input, "https://bilibili.com/playlist/BVlist");
    await userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /preview playlist/i })).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(api.previewPlaylistMetadata).toHaveBeenCalledWith(["BV1", "BV2"], "bilibili");
    });
  });
});

describe("SourcesListMode retry of a failed job", () => {
  it("surfaces the skip and dismisses the row when the video is already a source", async () => {
    // Dedup returns the video as already-processed: queued empty, skipped set.
    state.jobs = [makeFailedIngestJob()];
    state.ingestResult = { queued: [], skipped: ["BV1"] };
    renderMode();

    const retryBtn = await screen.findByRole("button", { name: /retry/i });
    await userEvent.click(retryBtn);

    await waitFor(() => {
      expect(screen.getByText(/already in this list/i)).toBeInTheDocument();
    });
    // The stale failed row is cleared rather than left as a silent no-op.
    expect(api.deleteJob).toHaveBeenCalledWith("job-failed-1");
  });

  it("queues a new job and dismisses the old when the video is retryable", async () => {
    state.jobs = [makeFailedIngestJob()];
    state.ingestResult = { queued: ["job-new"], skipped: [] };
    renderMode();

    const retryBtn = await screen.findByRole("button", { name: /retry/i });
    await userEvent.click(retryBtn);

    await waitFor(() => {
      expect(api.ingestUrl).toHaveBeenCalledWith(
        "list-1",
        expect.arrayContaining([expect.objectContaining({ video_id: "BV1" })]),
      );
    });
    expect(api.deleteJob).toHaveBeenCalledWith("job-failed-1");
  });
});

describe("SourcesListMode parent notification", () => {
  it("notifies the parent after a source is deleted", async () => {
    const onRefresh = vi.fn();
    renderMode([makeSource()], [], vi.fn(), onRefresh);

    await userEvent.click(screen.getByRole("button", { name: /source options/i }));
    await userEvent.click(screen.getByText(/^delete$/i));

    expect(api.deleteSource).toHaveBeenCalledWith("list-1", "src-1");
    await waitFor(() => {
      expect(onRefresh).toHaveBeenCalled();
    });
  });
});

describe("SourcesListMode source selection checkboxes", () => {
  const sources = [
    makeSource({ id: "src-1", title: "Source 1" }),
    makeSource({ id: "src-2", title: "Source 2" }),
    makeSource({ id: "src-3", title: "Source 3" }),
  ];

  it("select-all: all checked when all sources selected", () => {
    const onChange = vi.fn();
    renderMode(sources, ["src-1", "src-2", "src-3"], onChange);

    const selectAll = screen.getByRole("checkbox", { name: /select all/i });
    expect(selectAll).toBeChecked();
  });

  it("select-all: unchecked when no sources selected", () => {
    const onChange = vi.fn();
    renderMode(sources, [], onChange);

    const selectAll = screen.getByRole("checkbox", { name: /select all/i });
    expect(selectAll).not.toBeChecked();
  });

  it("select-all: indeterminate when partially selected", () => {
    const onChange = vi.fn();
    renderMode(sources, ["src-1"], onChange);

    const selectAll = screen.getByRole("checkbox", { name: /select all/i });
    expect(selectAll).not.toBeChecked();
    expect(selectAll).toHaveProperty("indeterminate", true);
  });

  it("select-all click: selects all when none selected", () => {
    const onChange = vi.fn();
    renderMode(sources, [], onChange);

    const selectAll = screen.getByRole("checkbox", { name: /select all/i });
    fireEvent.click(selectAll);

    expect(onChange).toHaveBeenCalledWith(["src-1", "src-2", "src-3"]);
  });

  it("select-all click: deselects all when all selected", () => {
    const onChange = vi.fn();
    renderMode(sources, ["src-1", "src-2", "src-3"], onChange);

    const selectAll = screen.getByRole("checkbox", { name: /select all/i });
    fireEvent.click(selectAll);

    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("row checkbox: toggles individual source", () => {
    const onChange = vi.fn();
    renderMode(sources, ["src-1", "src-2"], onChange);

    const rowCheckbox = screen.getByRole("checkbox", { name: /Select Source 3/i });
    fireEvent.click(rowCheckbox);

    expect(onChange).toHaveBeenCalledWith(["src-1", "src-2", "src-3"]);
  });
});
