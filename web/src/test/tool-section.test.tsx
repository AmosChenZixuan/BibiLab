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
    createApiClient: () => mockApi,
  };
});

// Stub the job activity context so trackJobs is observable in tests.
vi.mock("@/components/jobs/JobActivityProvider", async () => {
  const actual = await vi.importActual<typeof import("@/components/jobs/JobActivityProvider")>(
    "@/components/jobs/JobActivityProvider",
  );
  return {
    ...actual,
    useJobActivity: vi.fn(() => ({
      trackJobs: vi.fn(),
      jobs: [],
      dismissJob: vi.fn(),
      activeJobs: [],
    })),
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

  test("Mind Map card submits a mind_map job directly without a modal", async () => {
    const { api } = await import("@/lib/api");
    const { useJobActivity } = await import("@/components/jobs/JobActivityProvider");
    const trackJobs = vi.fn();
    vi.mocked(useJobActivity).mockReturnValue({
      trackJobs,
      activeJobs: [],
      visibleJobs: [],
      isPanelOpen: false,
      isPolling: false,
      isJobPending: () => false,
      errorMessage: null,
      clearTerminalJobs: vi.fn(),
      dismissJob: vi.fn(),
      cancelJob: vi.fn(),
      refreshNow: vi.fn(),
      setPanelOpen: vi.fn(),
      getJobs: () => [],
    });
    renderToolSection();
    const mindMapBtn = screen.getByRole("button", { name: /mind map/i });
    await userEvent.click(mindMapBtn);
    expect(api.createArtifact).toHaveBeenCalledWith(
      "list-1",
      expect.objectContaining({
        type: "mind_map",
        source_ids: ["src-1", "src-2"],
      }),
    );
    // trackJobs wires the job toast into the UI; a regression that
    // drops the call would make the job run silently.
    expect(trackJobs).toHaveBeenCalledWith([
      expect.objectContaining({ id: "job-1", producer: "artifact", label: "mind_map", contextKey: "list-1" }),
    ]);
    // No modal opens for the mind map — it's a direct action.
    expect(screen.queryByRole("heading", { name: "Generate Report" })).not.toBeInTheDocument();
  });

  test("Mind Map card is disabled when no sources are selected", () => {
    renderToolSection({ selectedSourceIds: [] });
    const mindMapBtn = screen.getByRole("button", { name: /mind map/i });
    expect(mindMapBtn).toBeDisabled();
  });

  test("Mind Map card is disabled while the createArtifact request is in flight", async () => {
    const { api } = await import("@/lib/api");
    let resolveCreate: (value: Awaited<ReturnType<typeof api.createArtifact>>) => void = () => {};
    vi.mocked(api.createArtifact).mockReturnValue(
      new Promise((resolve) => {
        resolveCreate = resolve as typeof resolveCreate;
      }) as ReturnType<typeof api.createArtifact>,
    );
    renderToolSection();
    const mindMapBtn = screen.getByRole("button", { name: /mind map/i });
    await userEvent.click(mindMapBtn);
    // The re-entrancy guard: while the request is pending, the button
    // must be disabled so a second click can't double-submit.
    expect(mindMapBtn).toBeDisabled();
    resolveCreate({
      id: "job-1",
      type: "ingest",
      status: "queued",
      progress: 0,
      error: null,
      created_at: "2026-04-08T00:00:00Z",
      updated_at: "2026-04-08T00:00:00Z",
      meta: {},
    } as Awaited<ReturnType<typeof api.createArtifact>>);
    // Drain microtasks so the finally block releases the disabled state.
    await vi.waitFor(() => expect(mindMapBtn).not.toBeDisabled());
  });
});
