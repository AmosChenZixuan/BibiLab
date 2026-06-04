import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { LabPanel } from "@/components/lists/LabPanel";
import { api } from "@/lib/api";
import { renderWithProviders } from "@/test/utils";

const ARTIFACT_1 = {
  id: "artifact-1",
  name: "Study Guide",
  type: "study_guide" as const,
  prompt: "Generate a study guide",
  source_ids: ["source-1", "source-2", "source-3"],
  status: "completed" as const,
  created_at: "2026-04-08T12:00:00Z",
};

vi.mock("@/lib/api", () => ({
  api: {
    getHealth: vi.fn().mockResolvedValue({}),
    listJobs: vi.fn().mockResolvedValue([]),
    listLists: vi.fn().mockResolvedValue([]),
    listSources: vi.fn().mockResolvedValue([]),
    ingestUrl: vi.fn().mockResolvedValue({ queued: [], skipped: [] }),
    deleteSource: vi.fn().mockResolvedValue(undefined),
    updateList: vi.fn().mockResolvedValue(undefined),
    getSource: vi.fn().mockResolvedValue(undefined),
    rerunDigest: vi.fn(),
    deleteJob: vi.fn(),
    createList: vi.fn(),
    putConfig: vi.fn(),
    listModels: vi.fn(),
    listArtifacts: vi.fn().mockResolvedValue([]),
    getArtifactContent: vi.fn().mockResolvedValue({ content: "# Study Guide\n\nContent here" }),
  },
  setCurrentLang: vi.fn(),
}));

function renderLabPanel(props?: Partial<React.ComponentProps<typeof LabPanel>>) {
  return renderWithProviders(
    <LabPanel
      listId="list-1"
      labCollapsed={false}
      labW={300}
      selectedSourceIds={[]}
      artifacts={[]}
      onArtifactsChange={vi.fn()}
      onToggleCollapse={vi.fn()}
      {...props}
    />,
    { providers: [JobActivityProvider, LanguageProvider] },
  );
}

describe("LabPanel", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  test("renders Lab header with font-serif text-lg text-ink", () => {
    renderLabPanel();
    const heading = screen.getByRole("heading", { name: /lab/i });
    expect(heading).toBeInTheDocument();
    expect(heading.className).toMatch(/font-serif/);
    expect(heading.className).toMatch(/text-lg/);
    expect(heading.className).toMatch(/text-ink/);
  });

  test("renders tool section with reports button in tool-list mode", () => {
    renderLabPanel({ selectedSourceIds: ["source-1"] });
    expect(screen.getByTestId("tool-section")).toBeInTheDocument();
    expect(screen.getByText("Reports")).toBeInTheDocument();
  });

  test("tool-list mode collapse button calls onToggleCollapse", async () => {
    const onToggleCollapse = vi.fn();
    renderLabPanel({ onToggleCollapse });
    const collapseBtn = screen.getByRole("button", { name: /collapse/i });
    await userEvent.click(collapseBtn);
    expect(onToggleCollapse).toHaveBeenCalledTimes(1);
  });

  test("collapsed mode shows narrow strip with expand button", () => {
    renderLabPanel({ labCollapsed: true });
    expect(screen.getByRole("button", { name: /expand/i })).toBeInTheDocument();
  });

  test("collapsed panel expand button calls onToggleCollapse", async () => {
    const onToggleCollapse = vi.fn();
    renderLabPanel({ labCollapsed: true, onToggleCollapse });
    await userEvent.click(screen.getByRole("button", { name: /expand/i }));
    expect(onToggleCollapse).toHaveBeenCalledTimes(1);
  });

  test("clicking done artifact card enters viewer mode with artifact name and source count", async () => {
    vi.mocked(api.listArtifacts).mockResolvedValue([ARTIFACT_1]);
    renderLabPanel({ artifacts: [ARTIFACT_1] });
    // Wait for artifact to load in ArtifactList
    await waitFor(() => {
      expect(screen.getByText("Study Guide")).toBeInTheDocument();
    });
    // Click the artifact card (clickable area is the inner cursor-pointer div)
    const artifactCard = screen.getByText("Study Guide").closest("[class*='cursor-pointer']") as HTMLElement;
    await userEvent.click(artifactCard);
    // Wait for viewer mode to render (check for minimize button which is unique to viewer)
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /minimize/i })).toBeInTheDocument();
    });
    // Verify viewer header content
    expect(screen.getByText(/based on 3 sources/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
  });

  test("copy button in viewer mode copies markdown to clipboard", async () => {
    vi.mocked(api.listArtifacts).mockResolvedValue([ARTIFACT_1]);
    renderLabPanel({ artifacts: [ARTIFACT_1] });
    await waitFor(() => {
      expect(screen.getByText("Study Guide")).toBeInTheDocument();
    });
    // Click the artifact card to enter viewer mode
    const artifactCard = screen.getByText("Study Guide").closest("[class*='cursor-pointer']") as HTMLElement;
    await userEvent.click(artifactCard);
    // Wait for viewer mode
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
    });
    const copyBtn = screen.getByRole("button", { name: /copy/i });
    // Mock clipboard (jsdom doesn't have clipboard API)
    const clipboardWrite = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: clipboardWrite },
      configurable: true,
    });
    await userEvent.click(copyBtn);
    expect(clipboardWrite).toHaveBeenCalledWith("# Study Guide\n\nContent here");
  });

  test("minimize button in viewer mode returns to tool-list showing artifact list", async () => {
    vi.mocked(api.listArtifacts).mockResolvedValue([ARTIFACT_1]);
    renderLabPanel({ artifacts: [ARTIFACT_1] });
    await waitFor(() => {
      expect(screen.getByText("Study Guide")).toBeInTheDocument();
    });
    // Enter viewer mode
    const artifactCard = screen.getByText("Study Guide").closest("[class*='cursor-pointer']") as HTMLElement;
    await userEvent.click(artifactCard);
    // Wait for viewer mode
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /minimize/i })).toBeInTheDocument();
    });
    // Click minimize
    await userEvent.click(screen.getByRole("button", { name: /minimize/i }));
    // Back to tool-list, should show artifact list and tool section
    await waitFor(() => {
      expect(screen.getByText("Study Guide")).toBeInTheDocument();
    });
    expect(screen.getByTestId("tool-section")).toBeInTheDocument();
  });

  test("viewer mode does not show collapse button (shows minimize instead)", async () => {
    vi.mocked(api.listArtifacts).mockResolvedValue([ARTIFACT_1]);
    renderLabPanel({ artifacts: [ARTIFACT_1] });
    await waitFor(() => {
      expect(screen.getByText("Study Guide")).toBeInTheDocument();
    });
    // Enter viewer mode
    const artifactCard = screen.getByText("Study Guide").closest("[class*='cursor-pointer']") as HTMLElement;
    await userEvent.click(artifactCard);
    // Wait for viewer mode
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /minimize/i })).toBeInTheDocument();
    });
    // Should show minimize, NOT collapse
    expect(screen.queryByRole("button", { name: /collapse/i })).not.toBeInTheDocument();
  });

  test("collapsing does NOT reset labMode - stays in viewer after expand", async () => {
    vi.mocked(api.listArtifacts).mockResolvedValue([ARTIFACT_1]);
    const onToggleCollapse = vi.fn();
    const { rerender } = renderLabPanel({ labCollapsed: false, artifacts: [ARTIFACT_1], onToggleCollapse });
    await waitFor(() => {
      expect(screen.getByText("Study Guide")).toBeInTheDocument();
    });
    // Enter viewer mode
    const artifactCard = screen.getByText("Study Guide").closest("[class*='cursor-pointer']") as HTMLElement;
    await userEvent.click(artifactCard);
    // Wait for viewer mode
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /minimize/i })).toBeInTheDocument();
    });
    // Simulate collapse
    rerender(
      <JobActivityProvider>
        <LanguageProvider>
          <LabPanel
            listId="list-1"
            labCollapsed={true}
            labW={300}
            selectedSourceIds={[]}
            artifacts={[ARTIFACT_1]}
            onArtifactsChange={vi.fn()}
            onToggleCollapse={onToggleCollapse}
          />
        </LanguageProvider>
      </JobActivityProvider>,
    );
    expect(screen.getByRole("button", { name: /expand/i })).toBeInTheDocument();
    // Simulate expand
    rerender(
      <JobActivityProvider>
        <LanguageProvider>
          <LabPanel
            listId="list-1"
            labCollapsed={false}
            labW={300}
            selectedSourceIds={[]}
            artifacts={[ARTIFACT_1]}
            onArtifactsChange={vi.fn()}
            onToggleCollapse={onToggleCollapse}
          />
        </LanguageProvider>
      </JobActivityProvider>,
    );
    // Should still be in viewer mode
    expect(screen.getByRole("button", { name: /minimize/i })).toBeInTheDocument();
  });
});
