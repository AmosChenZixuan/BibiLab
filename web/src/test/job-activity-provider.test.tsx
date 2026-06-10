import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { JobActivityProvider, useJobActivity } from "@/components/jobs/JobActivityProvider";
import { LanguageProvider } from "@/app/LanguageContext";
import type { IngestJob, Job } from "@/lib/types";
import { renderWithProviders } from "@/test/utils";

vi.mock("../lib/api", () => {
  const mockApi = {
    deleteJob: vi.fn(),
    listJobs: vi.fn(),
  };
  return {
    createApiClient: () => mockApi,
    api: mockApi,
    ApiError: class ApiError extends Error {
      status: number;
      constructor(status: number, detail: string) {
        super(detail);
        this.status = status;
      }
    },
    toErrorMessageWithT: (error: unknown, _t: (key: string) => string) =>
      error instanceof Error ? error.message : "Request failed",
  };
});

import { api } from "@/lib/api";

function renderProvider() {
  return renderWithProviders(<Probe />, {
    providers: [LanguageProvider, JobActivityProvider],
  });
}

function Probe() {
  const { dismissJob, errorMessage, visibleJobs } = useJobActivity();

  return (
    <div>
      <div aria-label="job-count">{visibleJobs.length}</div>
      <div aria-label="job-ids">{visibleJobs.map((item) => item.job.id).join(",")}</div>
      <div aria-label="error-message">{errorMessage ?? ""}</div>
      {visibleJobs.map((item) => (
        <button
          key={item.job.id}
          type="button"
          onClick={() => void dismissJob(item.job.id)}
        >
          Dismiss {item.job.id}
        </button>
      ))}
    </div>
  );
}

function makeIngestJob(overrides: Partial<IngestJob> = {}): Job {
  return {
    id: "job-1",
    type: "ingest",
    status: "failed",
    progress: 100,
    error: "boom",
    created_at: "2026-03-31T20:00:00Z",
    updated_at: "2026-03-31T20:01:00Z",
    meta: {
      list_id: "list-1",
      title: "Recovered Job",
      platform: "bilibili",
      source_url: "https://www.bilibili.com/video/BV1recovered",
    },
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("job activity provider", () => {
  test("rehydrates jobs from the backend on mount", async () => {
    vi.mocked(api.listJobs).mockResolvedValue([makeIngestJob()]);

    renderProvider();

    await waitFor(() => {
      expect(screen.getByLabelText("job-count")).toHaveTextContent("1");
    });
    expect(screen.getByLabelText("job-ids")).toHaveTextContent("job-1");
  });

  test("deduplicates a rehydrated job by id", async () => {
    vi.mocked(api.listJobs).mockResolvedValue([makeIngestJob(), makeIngestJob()]);

    renderProvider();

    await waitFor(() => {
      expect(screen.getByLabelText("job-count")).toHaveTextContent("1");
    });
  });

  test("dismissJob removes state only after a successful delete", async () => {
    vi.mocked(api.listJobs).mockResolvedValue([makeIngestJob()]);
    vi.mocked(api.deleteJob).mockResolvedValue(undefined);

    renderProvider();

    await screen.findByRole("button", { name: /dismiss job-1/i });
    await userEvent.click(screen.getByRole("button", { name: /dismiss job-1/i }));

    await waitFor(() => {
      expect(api.deleteJob).toHaveBeenCalledWith("job-1");
      expect(screen.getByLabelText("job-count")).toHaveTextContent("0");
    });
  });

  test("dismissJob keeps the job visible when delete fails", async () => {
    vi.mocked(api.listJobs).mockResolvedValue([makeIngestJob()]);
    vi.mocked(api.deleteJob).mockRejectedValue(new Error("Delete failed"));

    renderProvider();

    await screen.findByRole("button", { name: /dismiss job-1/i });
    await userEvent.click(screen.getByRole("button", { name: /dismiss job-1/i }));

    await waitFor(() => {
      expect(api.deleteJob).toHaveBeenCalledWith("job-1");
      expect(screen.getByLabelText("job-count")).toHaveTextContent("1");
      expect(screen.getByLabelText("error-message")).toHaveTextContent("Delete failed");
    });
  });
});
