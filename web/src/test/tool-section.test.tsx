import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ToolSection } from "@/components/lists/lab/ToolSection";

// Mock the api
vi.mock("@/lib/api", () => ({
  api: {
    createArtifact: vi.fn().mockResolvedValue({
      id: "job-1",
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
  toErrorMessage: (error: unknown) => (error instanceof Error ? error.message : "Request failed"),
  createApiClient: () => ({
    listJobs: vi.fn().mockResolvedValue([]),
    deleteJob: vi.fn().mockResolvedValue(undefined),
    createArtifact: vi.fn().mockResolvedValue({
      id: "job-1",
      type: "ingest",
      status: "queued",
      progress: 0,
      error: null,
      created_at: "2026-04-08T00:00:00Z",
      updated_at: "2026-04-08T00:00:00Z",
      meta: {},
    }),
  }),
}));

function renderToolSection(props?: Partial<React.ComponentProps<typeof ToolSection>>) {
  return render(
    <LanguageProvider>
      <JobActivityProvider>
        <ToolSection
          listId="list-1"
          sourceIds={["src-1", "src-2"]}
          onArtifactGenerated={vi.fn()}
          {...props}
        />
      </JobActivityProvider>
    </LanguageProvider>,
  );
}

describe("ToolSection", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  test("renders a grid layout", () => {
    renderToolSection();
    // ToolSection should exist and contain a grid
    const section = screen.getByTestId("tool-section");
    expect(section).toBeInTheDocument();
  });

  test("renders Reports card with icon and label", () => {
    renderToolSection();
    expect(screen.getByText("Reports")).toBeInTheDocument();
  });

  test("clicking Reports card opens ReportsModal", async () => {
    renderToolSection();
    const reportsCard = screen.getByRole("button", { name: /reports/i });
    await userEvent.click(reportsCard);
    // Modal should open with dialog role and Reports as heading
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Generate Report" })).toBeInTheDocument();
  });
});
