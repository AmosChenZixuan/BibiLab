import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ReportsModal } from "@/components/lists/lab/ReportsModal";
import { api } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  api: {
    createArtifact: vi.fn().mockResolvedValue({
      id: "job-artifact-1",
      type: "ingest",
      status: "queued",
      progress: 0,
      error: null,
      created_at: "2026-04-08T00:00:00Z",
      updated_at: "2026-04-08T00:00:00Z",
      meta: {},
    }),
    listJobs: vi.fn().mockResolvedValue([]),
    deleteJob: vi.fn().mockResolvedValue(undefined),
  },
}));

function renderReportsModal(props?: Partial<React.ComponentProps<typeof ReportsModal>>) {
  return render(
    <LanguageProvider>
      <JobActivityProvider>
        <ReportsModal
          listId="list-1"
          sourceIds={["src-1", "src-2"]}
          onClose={vi.fn()}
          open={true}
          onArtifactGenerated={vi.fn()}
          {...props}
        />
      </JobActivityProvider>
    </LanguageProvider>,
  );
}

describe("ReportsModal", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  test("renders modal with title 'Reports'", () => {
    renderReportsModal({ open: true });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Reports")).toBeInTheDocument();
  });

  test("renders 3 suggested prompt cards (Brief, Study Guide, Blog Post)", () => {
    renderReportsModal({ open: true });
    expect(screen.getByText("Brief")).toBeInTheDocument();
    expect(screen.getByText("Study Guide")).toBeInTheDocument();
    expect(screen.getByText("Blog Post")).toBeInTheDocument();
  });

  test("renders text input with placeholder", () => {
    renderReportsModal({ open: true });
    const input = screen.getByPlaceholderText(/describe what you need/i);
    expect(input).toBeInTheDocument();
  });

  test("clicking suggested card calls createArtifact and closes modal", async () => {
    const onClose = vi.fn();
    renderReportsModal({ open: true, onClose });

    const briefCard = screen.getByRole("button", { name: /brief/i });
    await userEvent.click(briefCard);

    await waitFor(() => {
      expect(api.createArtifact).toHaveBeenCalledWith("list-1", {
        type: "brief",
        prompt: "Brief",
        source_ids: ["src-1", "src-2"],
      });
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("typing custom prompt and submitting calls createArtifact with custom_report type", async () => {
    const onClose = vi.fn();
    renderReportsModal({ open: true, onClose });

    const input = screen.getByPlaceholderText(/describe what you need/i);
    await userEvent.type(input, "Give me a detailed analysis");

    const submitBtn = screen.getByRole("button", { name: /submit/i });
    await userEvent.click(submitBtn);

    await waitFor(() => {
      expect(api.createArtifact).toHaveBeenCalledWith("list-1", {
        type: "custom_report",
        prompt: "Give me a detailed analysis",
        source_ids: ["src-1", "src-2"],
      });
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("clicking outside modal calls onClose without submission", async () => {
    const onClose = vi.fn();
    renderReportsModal({ open: true, onClose });

    // The modal backdrop is the element with data-testid="modal-backdrop"
    const backdrop = screen.getByTestId("modal-backdrop");
    // Click on the backdrop (outside the dialog)
    await userEvent.click(backdrop);

    expect(api.createArtifact).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("modal is not rendered when open is false", () => {
    renderReportsModal({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
