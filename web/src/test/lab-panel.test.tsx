import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { LabPanel } from "@/components/lists/LabPanel";

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
    generateOverview: vi.fn(),
    putConfig: vi.fn(),
    downloadWhisperModel: vi.fn(),
    listWhisperModels: vi.fn(),
  },
}));

function renderLabPanel(props?: Partial<React.ComponentProps<typeof LabPanel>>) {
  return render(
    <LanguageProvider>
      <LabPanel
        listId="list-1"
        labCollapsed={false}
        labW={300}
        onToggleCollapse={vi.fn()}
        {...props}
      />
    </LanguageProvider>,
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

  test("renders tool-list mode content (Tool and Artifact sections) by default", () => {
    renderLabPanel();
    expect(screen.getByText(/tool/i)).toBeInTheDocument();
    expect(screen.getByText(/artifact/i)).toBeInTheDocument();
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

  test("minimize button in viewer mode returns to tool-list", async () => {
    const { rerender } = renderLabPanel();
    // Enter viewer mode by clicking the artifact button
    const artifactText = screen.getByText("Artifact");
    await userEvent.click(artifactText);
    // Now minimize button should appear
    expect(screen.getByRole("button", { name: /minimize/i })).toBeInTheDocument();
    // Click minimize to return to tool-list
    await userEvent.click(screen.getByRole("button", { name: /minimize/i }));
    // Should be back in tool-list mode with tool + artifact sections
    expect(screen.getByText(/tool/i)).toBeInTheDocument();
    expect(screen.getByText(/artifact/i)).toBeInTheDocument();
  });

  test("collapsing does NOT reset labMode - stays in viewer after expand", async () => {
    const onToggleCollapse = vi.fn();
    const { rerender } = renderLabPanel({ labCollapsed: false, onToggleCollapse });
    // Enter viewer mode by clicking artifact
    const artifactText = screen.getByText("Artifact");
    await userEvent.click(artifactText);
    // Verify minimize button visible (in viewer mode)
    expect(screen.getByRole("button", { name: /minimize/i })).toBeInTheDocument();
    // Simulate collapse by re-rendering with labCollapsed=true
    rerender(
      <LanguageProvider>
        <LabPanel
          listId="list-1"
          labCollapsed={true}
          labW={300}
          onToggleCollapse={onToggleCollapse}
        />
      </LanguageProvider>,
    );
    // In collapsed state, should show expand button
    expect(screen.getByRole("button", { name: /expand/i })).toBeInTheDocument();
    // Simulate expand by re-rendering with labCollapsed=false
    rerender(
      <LanguageProvider>
        <LabPanel
          listId="list-1"
          labCollapsed={false}
          labW={300}
          onToggleCollapse={onToggleCollapse}
        />
      </LanguageProvider>,
    );
    // Should still be in viewer mode (minimize button visible, artifact section gone)
    expect(screen.getByRole("button", { name: /minimize/i })).toBeInTheDocument();
  });
});
