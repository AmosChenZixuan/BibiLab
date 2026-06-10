import { act, render, screen, fireEvent, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { JobSpirit } from "@/components/jobs/JobSpirit";
import { LanguageProvider } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import { SourcesViewerMode } from "@/components/lists/sources/SourcesViewerMode";
import type { Job, Source, SourceContent } from "@/lib/types";

const { trackSpy } = vi.hoisted(() => ({ trackSpy: vi.fn() }));

vi.mock("@/components/jobs/JobActivityProvider", async () => {
  const actual = await vi.importActual<typeof import("@/components/jobs/JobActivityProvider")>(
    "@/components/jobs/JobActivityProvider",
  );
  return {
    ...actual,
    useJobActivity: () => {
      const ctx = actual.useJobActivity();
      return { ...ctx, trackJobs: trackSpy };
    },
  };
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  trackSpy.mockReset();
});

describe("SourcesViewerMode rerun", () => {
  test("rerun registers digest job with source title as label", async () => {
    vi.spyOn(api, "listJobs").mockResolvedValue([]);
    vi.spyOn(api, "rerunDigest").mockResolvedValue({ job_id: "new-job-1" });

    const source: Source = {
      id: "src-1",
      video_id: "BV1",
      platform: "bilibili",
      title: "Rerun Test Video",
      cover_url: null,
      source_url: "https://example.com",
      duration_seconds: 0,
      uploader: "",
      language: null,
      processed_at: "2026-01-01T00:00:00Z",
    };
    const content: SourceContent = {
      id: "src-1",
      video_id: "BV1",
      platform: "bilibili",
      title: "Rerun Test Video",
      source_url: "https://example.com",
      duration_seconds: 0,
      uploader: "",
      language: null,
      processed_at: "2026-01-01T00:00:00Z",
      cover_url: null,
      transcript: "",
      settings_snapshot: {},
    };

    render(
      <LanguageProvider>
        <JobActivityProvider>
          <SourcesViewerMode
            source={source}
            sourceContent={content}
            onRefresh={vi.fn()}
            listId="list-1"
          />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    fireEvent.click(screen.getByLabelText("Digest options"));
    fireEvent.click(screen.getByText("Re-run digest"));

    await act(async () => {
      await Promise.resolve();
    });

    expect(trackSpy).toHaveBeenCalledWith([
      expect.objectContaining({ id: "new-job-1", label: "Rerun Test Video" }),
    ]);
  });

  test("polled digest job (page refresh path) uses meta.source_title as label", async () => {
    const DIGEST_JOB_FROM_POLL: Job = {
      id: "polled-job",
      type: "digest",
      status: "queued",
      progress: 0,
      error: null,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      meta: { source_id: "src-1", list_id: "list-1", source_title: "Episode 47" },
    };
    vi.spyOn(api, "listJobs").mockResolvedValue([DIGEST_JOB_FROM_POLL]);

    render(
      <LanguageProvider>
        <JobActivityProvider>
          <JobSpirit />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    await act(async () => {
      await Promise.resolve();
    });

    // Open the spirit panel to see the row label.
    fireEvent.click(screen.getByLabelText("Jobs"));

    expect(screen.queryByText("Digest")).not.toBeInTheDocument();
    expect(screen.getByText("Episode 47")).toBeInTheDocument();
  });
});
