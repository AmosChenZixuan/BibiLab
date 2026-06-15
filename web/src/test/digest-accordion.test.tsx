import { act, render, screen, fireEvent, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { DigestAccordion } from "@/components/lists/DigestAccordion";
import { JobActivityProvider, useJobActivity } from "@/components/jobs/JobActivityProvider";
import { LanguageProvider } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import { TEST_IDS } from "@/lib/test-ids";
import type { Job } from "@/lib/types";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const baseProps = {
  source: { id: "src-1" },
  onRerun: vi.fn(),
  onRefresh: vi.fn(),
  facets: { seriesName: "罗翔说刑法", sequenceNumber: 8, seasonNumber: null },
  onSaveFacets: vi.fn().mockResolvedValue(undefined),
  listId: "list-1",
  sections: [
    { section_id: "sec-1", seq: 1, summary: "foo bar", keywords: ["alpha", "beta"], timestamp_start: 0, timestamp_end: 60 },
  ],
};

const DONE_DIGEST_JOB: Job = {
  id: "job-1",
  type: "digest",
  status: "done",
  progress: 100,
  error: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  meta: { source_id: "src-1", list_id: "list-1" },
};

// Forces JobActivityProvider to re-render (via setPanelOpen), which forces
// DigestAccordion to re-render and re-run its useEffect — same shape as a poll.
function PanelToggle() {
  const { setPanelOpen, isPanelOpen } = useJobActivity();
  return (
    <button
      type="button"
      data-testid="panel-toggle"
      onClick={() => setPanelOpen(!isPanelOpen)}
    >
      toggle
    </button>
  );
}

function withApiMocked() {
  vi.spyOn(api, "listJobs").mockResolvedValue([DONE_DIGEST_JOB]);
  return vi.spyOn(api, "deleteJob").mockResolvedValue(undefined);
}

describe("DigestAccordion", () => {
  test("renders keywords as chips", () => {
    render(
      <LanguageProvider>
        <JobActivityProvider>
          <DigestAccordion {...baseProps} />
        </JobActivityProvider>
      </LanguageProvider>,
    );
    expect(screen.getAllByText(/^(alpha|beta)$/)).toHaveLength(2);
  });

  test("keyword chips are clickable and fire onDiscussKeyword with the translated message", () => {
    const onDiscussKeyword = vi.fn();
    render(
      <LanguageProvider>
        <JobActivityProvider>
          <DigestAccordion {...baseProps} onDiscussKeyword={onDiscussKeyword} />
        </JobActivityProvider>
      </LanguageProvider>,
    );
    const chips = screen.getAllByTestId(TEST_IDS.digestKeywordChip);
    expect(chips).toHaveLength(2);
    expect(chips[0]).toHaveTextContent("alpha");
    expect(chips[0]).toHaveAttribute("aria-label", "Discuss alpha");
    fireEvent.click(chips[0]);
    expect(onDiscussKeyword).toHaveBeenCalledTimes(1);
    expect(onDiscussKeyword).toHaveBeenCalledWith("Discuss alpha");
    fireEvent.click(chips[1]);
    expect(onDiscussKeyword).toHaveBeenCalledTimes(2);
    expect(onDiscussKeyword).toHaveBeenLastCalledWith("Discuss beta");
  });

  test("shows facet strip and an Edit metadata menu item", () => {
    render(
      <LanguageProvider>
        <JobActivityProvider>
          <DigestAccordion {...baseProps} />
        </JobActivityProvider>
      </LanguageProvider>,
    );
    expect(screen.getByText(/罗翔说刑法/)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Digest options"));
    expect(screen.getByText("Edit metadata")).toBeInTheDocument();
  });

  test("auto-dismisses a done digest job (no delay) and refetches the source", async () => {
    const deleteJob = withApiMocked();

    render(
      <LanguageProvider>
        <JobActivityProvider>
          <DigestAccordion {...baseProps} />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    // Flush initial listJobs fetch + auto-dismiss chain
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(deleteJob).toHaveBeenCalledWith("job-1");
    expect(baseProps.onRefresh).toHaveBeenCalledWith("src-1");
  });
});
