import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ReportsModal } from "@/components/lists/lab/ReportsModal";
import { api } from "@/lib/api";
import { renderWithProviders } from "@/test/utils";

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
  setCurrentLang: vi.fn(),
}));

function renderReportsModal(props?: Partial<React.ComponentProps<typeof ReportsModal>>) {
  return renderWithProviders(
    <ReportsModal
      listId="list-1"
      sourceIds={["src-1", "src-2"]}
      onClose={vi.fn()}
      open={true}
      {...props}
    />,
    { providers: [LanguageProvider, JobActivityProvider] },
  );
}

describe("ReportsModal", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  test("renders modal with title 'Generate Report'", () => {
    renderReportsModal({ open: true });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Generate Report")).toBeInTheDocument();
  });

  test("renders 4 format options (Brief, Study Guide, Blog Post, Custom)", () => {
    renderReportsModal({ open: true });
    expect(screen.getByText("Brief")).toBeInTheDocument();
    expect(screen.getByText("Study Guide")).toBeInTheDocument();
    expect(screen.getByText("Blog Post")).toBeInTheDocument();
    expect(screen.getByText("Custom")).toBeInTheDocument();
  });

  test("renders textarea with prompt placeholder when custom is selected", () => {
    renderReportsModal({ open: true });
    // Custom is selected by default, shows prompt placeholder
    expect(screen.getByPlaceholderText(/summarize the key arguments/i)).toBeInTheDocument();
  });

  test("clicking Brief format fills textarea with template", async () => {
    renderReportsModal({ open: true });

    const briefBtn = screen.getByRole("button", { name: /brief/i });
    await userEvent.click(briefBtn);

    // Textarea should now contain the Brief template
    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea.value).toContain("brief");
  });

  test("clicking format then submit calls createArtifact with format type", async () => {
    const onClose = vi.fn();
    renderReportsModal({ open: true, onClose });

    // Select Brief format
    const briefBtn = screen.getByRole("button", { name: /brief/i });
    await userEvent.click(briefBtn);

    // Submit
    const submitBtn = screen.getByRole("button", { name: /submit/i });
    await userEvent.click(submitBtn);

    await waitFor(() => {
      expect(api.createArtifact).toHaveBeenCalledWith("list-1", {
        type: "brief",
        prompt: expect.stringContaining("Create a comprehensive briefing document"),
        source_ids: ["src-1", "src-2"],
      });
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("custom format is selected by default and textarea is empty", async () => {
    const onClose = vi.fn();
    renderReportsModal({ open: true, onClose });

    // Custom is selected by default, textarea should be empty
    const textarea = screen.getByRole("textbox");
    expect(textarea).toHaveValue("");

    // Type a custom prompt
    await userEvent.type(textarea, "Give me a detailed analysis");

    // Submit
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
