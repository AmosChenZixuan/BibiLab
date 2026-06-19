// MindMapBlock interactive tree viewer. Renders a JSON ``` fence as
// an expandable horizontal tree.

import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { ArtifactViewer } from "@/components/lists/lab/ArtifactViewer";
import { renderWithProviders } from "@/test/utils";

vi.mock("@/lib/api", async () => {
  const { createMockApi } = await import("@/test/utils");
  // Generic structural example — no domain-specific content, so the
  // tree shape alone is under test.
  const mindMapContent = [
    "```json",
    "{",
    '  "name": "Sample Topic",',
    '  "root": {',
    '    "label": "Main Topic",',
    '    "children": [',
    '      {"label": "Branch A", "children": [',
    '        {"label": "Leaf A1"},',
    '        {"label": "Leaf A2"}',
    "      ]},",
    '      {"label": "Branch B", "children": [',
    '        {"label": "Leaf B1"}',
    "      ]}",
    "    ]",
    "  }",
    "}",
    "```",
    "",
  ].join("\n");
  const mock = createMockApi({
    getArtifactContent: vi.fn().mockResolvedValue({ content: mindMapContent }),
  });
  return { api: mock, createApiClient: () => mock };
});

const MIND_MAP_ARTIFACT = {
  id: "art-mm",
  name: "Sample Topic",
  type: "mind_map" as const,
  prompt: "ignored",
  source_ids: ["src-1", "src-2", "src-3", "src-4"],
  status: "completed" as const,
  created_at: "2026-06-19T00:00:00Z",
};

function renderArtifact(artifact = MIND_MAP_ARTIFACT) {
  return renderWithProviders(<ArtifactViewer artifact={artifact} />, {
    providers: [LanguageProvider],
  });
}

describe("ArtifactViewer mind map", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  test("renders root + branches expanded, leaves collapsed by default", async () => {
    renderArtifact();
    expect(await screen.findByTestId("mindmap-canvas")).toBeInTheDocument();
    expect(screen.getByText("Main Topic")).toBeInTheDocument();
    expect(screen.getByText("Branch A")).toBeInTheDocument();
    expect(screen.getByText("Branch B")).toBeInTheDocument();
    expect(screen.queryByText("Leaf A1")).not.toBeInTheDocument();
    expect(screen.queryByText("Leaf A2")).not.toBeInTheDocument();
    expect(screen.queryByText("Leaf B1")).not.toBeInTheDocument();
  });

  test("clicking a node's toggle button expands its subtree; clicking again collapses it", async () => {
    renderArtifact();
    await screen.findByTestId("mindmap-canvas");
    const toggleBtn = screen.getByTestId("mindmap-toggle-0.0");
    // Expand — leaves of Branch A appear; sibling branch leaves stay hidden.
    await userEvent.click(toggleBtn);
    expect(await screen.findByText("Leaf A1")).toBeInTheDocument();
    expect(screen.getByText("Leaf A2")).toBeInTheDocument();
    expect(screen.queryByText("Leaf B1")).not.toBeInTheDocument();
    // Collapse — leaves disappear.
    await userEvent.click(toggleBtn);
    expect(screen.queryByText("Leaf A1")).not.toBeInTheDocument();
  });

  test("shows error banner when content has no JSON fence", async () => {
    const { api } = await import("@/lib/api");
    vi.mocked(api.getArtifactContent).mockResolvedValueOnce({
      content: "# no fence here",
    });
    renderArtifact();
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/malformed or could not be parsed/)).toBeInTheDocument();
  });
});
