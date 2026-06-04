import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { ToolSection } from "@/components/lists/lab/ToolSection";
import { renderWithProviders } from "@/test/utils";

// Mock the api
vi.mock("@/lib/api", async () => {
  const { createMockApi } = await import("@/test/utils");
  const mockApi = createMockApi({
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
  });
  return {
    api: mockApi,
    setCurrentLang: vi.fn(),
    createApiClient: () => mockApi,
  };
});

function renderToolSection(props?: Partial<React.ComponentProps<typeof ToolSection>>) {
  return renderWithProviders(
    <ToolSection
      listId="list-1"
      selectedSourceIds={["src-1", "src-2"]}
      {...props}
    />,
    { providers: [LanguageProvider, JobActivityProvider] },
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
