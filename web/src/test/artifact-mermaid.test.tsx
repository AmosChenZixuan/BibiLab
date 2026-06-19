// MindMapBlock interactive tree viewer. Renders a JSON ``` fence as
// an expandable vertical tree with pan/zoom controls.

import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { ArtifactViewer } from "@/components/lists/lab/ArtifactViewer";
import { renderWithProviders, createMockApi } from "@/test/utils";

vi.mock("@/lib/api", async () => {
  const { createMockApi } = await import("@/test/utils");
  const mindMapContent = [
    "# 美食菜谱层次图",
    "",
    "```json",
    "{",
    '  "name": "美食菜谱层次图",',
    '  "root": {',
    '    "label": "美食菜谱",',
    '    "children": [',
    '      {"label": "中式佳肴", "children": [',
    '        {"label": "红油冒菜"},',
    '        {"label": "鸡公煲"}',
    "      ]},",
    '      {"label": "西式主菜", "children": [',
    '        {"label": "黛安娜牛排"}',
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
  name: "美食菜谱层次图",
  type: "mind_map" as const,
  prompt: "ignored",
  source_ids: ["src-1", "src-2", "src-3", "src-4"],
  status: "completed" as const,
  created_at: "2026-06-19T00:00:00Z",
};

describe("ArtifactViewer mind map", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  test("renders the root + branches collapsed by default (leaves hidden)", async () => {
    renderWithProviders(<ArtifactViewer artifact={MIND_MAP_ARTIFACT} />, {
      providers: [LanguageProvider],
    });
    expect(await screen.findByTestId("mindmap-canvas")).toBeInTheDocument();
    // Root and direct branches are visible.
    expect(screen.getByText("美食菜谱")).toBeInTheDocument();
    expect(screen.getByText("中式佳肴")).toBeInTheDocument();
    expect(screen.getByText("西式主菜")).toBeInTheDocument();
    // Leaves (depth >= 2) are NOT visible until the parent branch expands.
    expect(screen.queryByText("红油冒菜")).not.toBeInTheDocument();
    expect(screen.queryByText("鸡公煲")).not.toBeInTheDocument();
    expect(screen.queryByText("黛安娜牛排")).not.toBeInTheDocument();
  });

  test("clicking the expand button reveals a branch's leaves", async () => {
    renderWithProviders(<ArtifactViewer artifact={MIND_MAP_ARTIFACT} />, {
      providers: [LanguageProvider],
    });
    await screen.findByTestId("mindmap-canvas");
    const toggleBtn = screen.getByTestId("mindmap-toggle-0.0");
    await userEvent.click(toggleBtn);
    expect(await screen.findByText("红油冒菜")).toBeInTheDocument();
    expect(screen.getByText("鸡公煲")).toBeInTheDocument();
    // Sibling branch's leaves are still hidden.
    expect(screen.queryByText("黛安娜牛排")).not.toBeInTheDocument();
  });

  test("clicking the expand button again collapses the branch", async () => {
    renderWithProviders(<ArtifactViewer artifact={MIND_MAP_ARTIFACT} />, {
      providers: [LanguageProvider],
    });
    await screen.findByTestId("mindmap-canvas");
    const toggleBtn = screen.getByTestId("mindmap-toggle-0.0");
    await userEvent.click(toggleBtn);
    expect(await screen.findByText("红油冒菜")).toBeInTheDocument();
    await userEvent.click(toggleBtn);
    await waitFor(() => {
      expect(screen.queryByText("红油冒菜")).not.toBeInTheDocument();
    });
  });

  test("clicking the node card itself does NOT toggle expand (it's a separate target)", async () => {
    renderWithProviders(<ArtifactViewer artifact={MIND_MAP_ARTIFACT} />, {
      providers: [LanguageProvider],
    });
    await screen.findByTestId("mindmap-canvas");
    // Click the 中式佳肴 node card (role=button, separate from the toggle).
    const branchCard = screen.getByRole("button", { name: "中式佳肴" });
    await userEvent.click(branchCard);
    // Clicking the card alone must not collapse or expand.
    expect(screen.queryByText("红油冒菜")).not.toBeInTheDocument();
  });

  test("renders pan/zoom controls", async () => {
    renderWithProviders(<ArtifactViewer artifact={MIND_MAP_ARTIFACT} />, {
      providers: [LanguageProvider],
    });
    expect(await screen.findByTestId("mindmap-canvas")).toBeInTheDocument();
    expect(screen.getByLabelText("Zoom in")).toBeInTheDocument();
    expect(screen.getByLabelText("Zoom out")).toBeInTheDocument();
    expect(screen.getByLabelText("Reset view")).toBeInTheDocument();
  });

  test("shows error banner when content has no JSON fence", async () => {
    const { api } = await import("@/lib/api");
    vi.mocked(api.getArtifactContent).mockResolvedValueOnce({
      content: "# no fence here",
    });

    renderWithProviders(<ArtifactViewer artifact={MIND_MAP_ARTIFACT} />, {
      providers: [LanguageProvider],
    });
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/Mind map data is malformed/)).toBeInTheDocument();
  });

  test("falls back to Mermaid flowchart parser for old-format artifacts", async () => {
    const { api } = await import("@/lib/api");
    const oldMermaid = [
      "```mermaid",
      "flowchart TD",
      "A[Root Label]",
      "A --> B[Branch]",
      "B --> C[Leaf]",
      "```",
    ].join("\n");
    vi.mocked(api.getArtifactContent).mockResolvedValueOnce({
      content: oldMermaid,
    });

    renderWithProviders(<ArtifactViewer artifact={MIND_MAP_ARTIFACT} />, {
      providers: [LanguageProvider],
    });
    expect(await screen.findByTestId("mindmap-canvas")).toBeInTheDocument();
    expect(screen.getByText("Root Label")).toBeInTheDocument();
    // Branches are collapsed by default — verify the branch is there.
    expect(screen.getByText("Branch")).toBeInTheDocument();
    // Leaf is at depth 2, so it's hidden until the branch expands.
    expect(screen.queryByText("Leaf")).not.toBeInTheDocument();
  });
});
