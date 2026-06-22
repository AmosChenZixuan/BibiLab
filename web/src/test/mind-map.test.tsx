// MindMapBlock interactive tree viewer. Renders a JSON ``` fence as
// an expandable horizontal tree.

import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { LanguageProvider } from "@/app/LanguageContext";
import { ArtifactViewer } from "@/components/lists/lab/ArtifactViewer";
import { TEST_IDS } from "@/lib/test-ids";
import type { Source } from "@/lib/types";
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

const SAMPLE_SOURCES: Source[] = [
  { id: "src-1", video_id: "v1", platform: "bilibili", title: "First Source", cover_url: null, source_url: "", duration_seconds: 0, uploader: "", language: null, processed_at: "" },
  { id: "src-2", video_id: "v2", platform: "bilibili", title: "Second Source", cover_url: null, source_url: "", duration_seconds: 0, uploader: "", language: null, processed_at: "" },
  { id: "src-3", video_id: "v3", platform: "bilibili", title: "Third Source", cover_url: null, source_url: "", duration_seconds: 0, uploader: "", language: null, processed_at: "" },
  // src-4 was deleted after the mindmap was generated — popover should
  // show a placeholder row to keep the row count aligned with the pill.
  { id: "src-5", video_id: "v5", platform: "bilibili", title: "Unrelated", cover_url: null, source_url: "", duration_seconds: 0, uploader: "", language: null, processed_at: "" },
];

function renderArtifact(
  artifact = MIND_MAP_ARTIFACT,
  props: { sources?: Source[]; onAskInChatFromMindmap?: (topic: string, parentTopic: string | null, sourceIds: string[]) => void; onOpenSource?: (s: Source) => void } = {},
) {
  return renderWithProviders(
    <ArtifactViewer artifact={artifact} sources={props.sources} onAskInChatFromMindmap={props.onAskInChatFromMindmap} onOpenSource={props.onOpenSource} />,
    { providers: [LanguageProvider] },
  );
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

  test("clicking a node fires onAskInChat with topic + parent label", async () => {
    const onAsk = vi.fn();
    renderArtifact(MIND_MAP_ARTIFACT, { onAskInChatFromMindmap: onAsk });
    await screen.findByTestId("mindmap-canvas");
    // Branch A is at depth 1, so its immediate parent is the root.
    await userEvent.click(screen.getByRole("button", { name: "Branch A" }));
    expect(onAsk).toHaveBeenCalledTimes(1);
    expect(onAsk).toHaveBeenCalledWith("Branch A", "Main Topic", MIND_MAP_ARTIFACT.source_ids);
  });

  test("clicking a leaf fires onAskInChat with the branch as parent (verifies recursive parentLabel threading)", async () => {
    const onAsk = vi.fn();
    renderArtifact(MIND_MAP_ARTIFACT, { onAskInChatFromMindmap: onAsk });
    await screen.findByTestId("mindmap-canvas");
    // Expand Branch A to reveal its leaves.
    await userEvent.click(screen.getByTestId("mindmap-toggle-0.0"));
    expect(await screen.findByText("Leaf A1")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Leaf A1" }));
    expect(onAsk).toHaveBeenCalledTimes(1);
    expect(onAsk).toHaveBeenCalledWith("Leaf A1", "Branch A", MIND_MAP_ARTIFACT.source_ids);
  });

  test("clicking the root node fires onAskInChat with parent=null", async () => {
    const onAsk = vi.fn();
    renderArtifact(MIND_MAP_ARTIFACT, { onAskInChatFromMindmap: onAsk });
    await screen.findByTestId("mindmap-canvas");
    await userEvent.click(screen.getByRole("button", { name: "Main Topic" }));
    expect(onAsk).toHaveBeenCalledTimes(1);
    expect(onAsk).toHaveBeenCalledWith("Main Topic", null, MIND_MAP_ARTIFACT.source_ids);
  });

  test("'View N sources' button opens a popover listing every artifact source_id, with deleted sources as placeholders", async () => {
    renderArtifact(MIND_MAP_ARTIFACT, { sources: SAMPLE_SOURCES });
    // The artifact has 4 source_ids: src-1/2/3 exist, src-4 was deleted.
    const pill = await screen.findByRole("button", { name: /view 4 sources/i });
    expect(pill).toBeInTheDocument();
    await userEvent.click(pill);
    // Existing sources render as buttons; the missing src-4 renders as a placeholder.
    expect(screen.getByText("First Source")).toBeInTheDocument();
    expect(screen.getByText("Second Source")).toBeInTheDocument();
    expect(screen.getByText("Third Source")).toBeInTheDocument();
    // Unrelated src-5 is NOT in the artifact's source_ids → must not appear.
    expect(screen.queryByText("Unrelated")).not.toBeInTheDocument();
    // The deleted src-4 slot shows the localized placeholder, NOT a filter drop.
    expect(screen.getByTestId(TEST_IDS.sourceRowDeleted)).toBeInTheDocument();
    expect(screen.getByText(/deleted source/i)).toBeInTheDocument();
    // 3 real rows + 1 deleted placeholder = 4 total = artifact.source_ids.length.
    expect(screen.getAllByTestId(TEST_IDS.sourceRow)).toHaveLength(3);
    expect(screen.getAllByTestId(TEST_IDS.sourceRowDeleted)).toHaveLength(1);
  });

  test("popover closes when Escape is pressed", async () => {
    renderArtifact(MIND_MAP_ARTIFACT, { sources: SAMPLE_SOURCES });
    const pill = await screen.findByRole("button", { name: /view 4 sources/i });
    await userEvent.click(pill);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("popover closes on outside click", async () => {
    renderArtifact(MIND_MAP_ARTIFACT, { sources: SAMPLE_SOURCES });
    const pill = await screen.findByRole("button", { name: /view 4 sources/i });
    await userEvent.click(pill);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await userEvent.click(document.body);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("clicking a real source row calls onOpenSource and closes the popover", async () => {
    const onOpen = vi.fn();
    renderArtifact(MIND_MAP_ARTIFACT, { sources: SAMPLE_SOURCES, onOpenSource: onOpen });
    const pill = await screen.findByRole("button", { name: /view 4 sources/i });
    await userEvent.click(pill);
    await userEvent.click(screen.getByText("First Source"));
    expect(onOpen).toHaveBeenCalledTimes(1);
    expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ id: "src-1", title: "First Source" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("when sources prop is undefined, every popover row renders as the deleted-source placeholder", async () => {
    renderArtifact(MIND_MAP_ARTIFACT);
    const pill = await screen.findByRole("button", { name: /view 4 sources/i });
    await userEvent.click(pill);
    // 4 artifact source_ids, none resolvable → 4 placeholder rows.
    expect(screen.getAllByTestId(TEST_IDS.sourceRowDeleted)).toHaveLength(4);
    expect(screen.queryByTestId(TEST_IDS.sourceRow)).not.toBeInTheDocument();
  });
});
