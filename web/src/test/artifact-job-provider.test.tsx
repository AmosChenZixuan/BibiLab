import { cleanup, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { JobActivityProvider, useJobActivity } from "@/components/jobs/JobActivityProvider";
import type { ArtifactJob, Job } from "@/lib/types";
import { renderWithProviders } from "@/test/utils";

vi.mock("@/lib/api", async () => {
  const { createMockApi } = await import("@/test/utils");
  return {
    createApiClient: () =>
      createMockApi({
        deleteJob: vi.fn(),
        listJobs: vi.fn(),
      }),
    api: createMockApi({
      deleteJob: vi.fn(),
      listJobs: vi.fn(),
    }),
  };
});

import { api } from "@/lib/api";

function renderProvider() {
  return renderWithProviders(<Probe />, {
    providers: [JobActivityProvider],
  });
}

function Probe() {
  const { getJobs } = useJobActivity();

  const artifactJobs = getJobs("artifact", "list-1");

  return (
    <div>
      <div aria-label="artifact-job-count">{artifactJobs.length}</div>
      {artifactJobs.map((item) => (
        <div key={item.job.id} aria-label={`artifact-job-${item.job.id}`}>
          {item.label} - {item.job.status}
        </div>
      ))}
    </div>
  );
}

function makeArtifactJob(overrides: Partial<ArtifactJob> = {}): Job {
  return {
    id: "artifact-job-1",
    type: "artifact",
    status: "generating",
    progress: 0,
    error: null,
    created_at: "2026-04-08T12:00:00Z",
    updated_at: "2026-04-08T12:00:00Z",
    meta: {
      list_id: "list-1",
      artifact_id: "artifact-1",
      artifact_type: "brief",
    },
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("artifact job support in JobActivityProvider", () => {
  test("getJobs returns artifact jobs for correct producer and contextKey", async () => {
    vi.mocked(api.listJobs).mockResolvedValue([makeArtifactJob()]);

    renderProvider();

    await waitFor(() => {
      expect(screen.getByLabelText("artifact-job-count")).toHaveTextContent("1");
    });
    expect(screen.getByLabelText("artifact-job-artifact-job-1")).toHaveTextContent("generating");
  });

  test("getJobs with artifact producer filters correctly", async () => {
    const artifactJob = makeArtifactJob({ id: "artifact-job-2" });
    vi.mocked(api.listJobs).mockResolvedValue([artifactJob]);

    renderProvider();

    await waitFor(() => {
      expect(screen.getByLabelText("artifact-job-count")).toHaveTextContent("1");
    });
  });
});
