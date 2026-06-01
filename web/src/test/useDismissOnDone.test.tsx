import { act, render } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { JobActivityProvider, useJobActivity } from "@/components/jobs/JobActivityProvider";
import { useDismissOnDone } from "@/components/jobs/useDismissOnDone";
import { LanguageProvider } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import type { Job } from "@/lib/types";

afterEach(() => {
  vi.restoreAllMocks();
});

const DONE_INGEST_JOB: Job = {
  id: "job-1",
  type: "ingest",
  status: "done",
  progress: 100,
  error: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  meta: { list_id: "list-1", title: "src", source_url: "x", platform: "local" },
};

const FAILED_INGEST_JOB: Job = {
  ...DONE_INGEST_JOB,
  id: "job-2",
  status: "failed",
  error: "boom",
};

function Harness({ onDone }: { onDone: (job: Job) => Promise<void> | void }) {
  const { getJobs } = useJobActivity();
  const jobs = getJobs("ingest", "list-1");
  useDismissOnDone({ jobs, onDone });
  return null;
}

function PollSimulator() {
  const { setPanelOpen, isPanelOpen } = useJobActivity();
  return (
    <button data-testid="poll" onClick={() => setPanelOpen(!isPanelOpen)}>
      poll
    </button>
  );
}

async function flush() {
  await act(async () => {
    await Promise.resolve();
  });
  await act(async () => {
    await Promise.resolve();
  });
}

describe("useDismissOnDone", () => {
  test("invokes onDone and dismisses terminal-done jobs", async () => {
    vi.spyOn(api, "listJobs").mockResolvedValue([DONE_INGEST_JOB]);
    const deleteJob = vi.spyOn(api, "deleteJob").mockResolvedValue(undefined);
    const onDone = vi.fn().mockResolvedValue(undefined);

    render(
      <LanguageProvider>
        <JobActivityProvider>
          <Harness onDone={onDone} />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    await flush();

    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ id: "job-1" }));
    expect(deleteJob).toHaveBeenCalledWith("job-1");
  });

  test("leaves failed jobs alone (no onDone, no dismiss)", async () => {
    vi.spyOn(api, "listJobs").mockResolvedValue([FAILED_INGEST_JOB]);
    const deleteJob = vi.spyOn(api, "deleteJob").mockResolvedValue(undefined);
    const onDone = vi.fn().mockResolvedValue(undefined);

    render(
      <LanguageProvider>
        <JobActivityProvider>
          <Harness onDone={onDone} />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    await flush();

    expect(onDone).not.toHaveBeenCalled();
    expect(deleteJob).not.toHaveBeenCalled();
  });

  test("onDone failure does not block dismissJob", async () => {
    vi.spyOn(api, "listJobs").mockResolvedValue([DONE_INGEST_JOB]);
    const deleteJob = vi.spyOn(api, "deleteJob").mockResolvedValue(undefined);
    const onDone = vi.fn().mockRejectedValue(new Error("refetch failed"));

    render(
      <LanguageProvider>
        <JobActivityProvider>
          <Harness onDone={onDone} />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    await flush();

    expect(onDone).toHaveBeenCalled();
    expect(deleteJob).toHaveBeenCalledWith("job-1");
  });

  test("re-renders do not re-trigger onDone or dismiss (ref dedup)", async () => {
    // Regression: poll cadence (~5s) re-emits the jobs array with a new identity
    // every tick. Without sync ref-based dedup, a cleanup-on-rerun pattern
    // could preempt the dismiss, or a state-based dedup could let stale jobs
    // re-trigger the chain.
    vi.spyOn(api, "listJobs").mockResolvedValue([DONE_INGEST_JOB]);
    const deleteJob = vi.spyOn(api, "deleteJob").mockResolvedValue(undefined);
    const onDone = vi.fn().mockResolvedValue(undefined);

    const { getByTestId } = render(
      <LanguageProvider>
        <JobActivityProvider>
          <PollSimulator />
          <Harness onDone={onDone} />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    await flush();
    const onDoneCallsAfterFirst = onDone.mock.calls.length;
    const deleteCallsAfterFirst = deleteJob.mock.calls.length;

    // Simulate several poll cycles
    await act(async () => {
      getByTestId("poll").click();
      getByTestId("poll").click();
      getByTestId("poll").click();
    });
    await flush();

    expect(onDone.mock.calls.length).toBe(onDoneCallsAfterFirst);
    expect(deleteJob.mock.calls.length).toBe(deleteCallsAfterFirst);
  });

  test("onDone may be a fresh function each render (ref captures latest)", async () => {
    vi.spyOn(api, "listJobs").mockResolvedValue([DONE_INGEST_JOB]);
    vi.spyOn(api, "deleteJob").mockResolvedValue(undefined);

    function UnstableHarness() {
      const [version, setVersion] = useState(0);
      useDismissOnDone({
        jobs: useJobActivity().getJobs("ingest", "list-1"),
        onDone: () => setVersion((v) => v + 1),
      });
      return <span data-testid="v">{version}</span>;
    }

    const { getByTestId } = render(
      <LanguageProvider>
        <JobActivityProvider>
          <UnstableHarness />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    await flush();

    // onDone fired once → version incremented once
    expect(getByTestId("v").textContent).toBe("1");
  });
});
