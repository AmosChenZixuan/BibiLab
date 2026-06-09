import { act, render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { LanguageProvider } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import { SourcesViewerMode } from "@/components/lists/sources/SourcesViewerMode";
import type { Source, SourceContent, SourceSection } from "@/lib/types";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function makeSource(id: string, title: string): Source {
  return {
    id,
    video_id: id,
    platform: "bilibili",
    title,
    summary: "",
    keywords: [],
    cover_url: null,
    source_url: "https://example.com",
    duration_seconds: 0,
    uploader: "",
    language: null,
    processed_at: "2026-01-01T00:00:00Z",
  };
}

function makeContent(id: string, title: string, summary: string): SourceContent {
  return {
    id,
    video_id: id,
    platform: "bilibili",
    title,
    source_url: "https://example.com",
    duration_seconds: 0,
    uploader: "",
    language: null,
    processed_at: "2026-01-01T00:00:00Z",
    summary,
    keywords: [],
    cover_url: null,
    transcript: "transcript text",
    settings_snapshot: {},
  };
}

function makeSections(n: number, offset = 0): SourceSection[] {
  return Array.from({ length: n }, (_, i) => ({
    seq: i + 1,
    summary: `Section ${offset + i + 1} summary`,
    keywords: [`kw-${offset + i + 1}`],
    timestamp_start: i * 600,
    timestamp_end: (i + 1) * 600,
  }));
}

describe("SourcesViewerMode sections — source-switch desync", () => {
  test("switching source.id resets the active section to the first one", async () => {
    // Source A has 5 sections; the test will navigate to section 3, then
    // switch to source B with 4 sections and assert that the visible
    // body is section 1 of source B (not section 3 of source B, which
    // is what would happen if activeSectionIdx persisted across the
    // remount).
    const contentA = makeContent("src-A", "Source A", "Source A summary");
    const contentB = makeContent("src-B", "Source B", "Source B summary");
    const sectionsA = makeSections(5, 0);
    const sectionsB = makeSections(4, 100);

    vi.spyOn(api, "getSource").mockImplementation(async (id) => {
      if (id === "src-A") return contentA;
      if (id === "src-B") return contentB;
      return undefined;
    });
    vi.spyOn(api, "getSourceSections").mockImplementation(async (id) => {
      if (id === "src-A") return sectionsA;
      if (id === "src-B") return sectionsB;
      return [];
    });
    vi.spyOn(api, "listJobs").mockResolvedValue([]);

    const sourceA = makeSource("src-A", "Source A");
    const sourceB = makeSource("src-B", "Source B");

    const renderViewer = (source: Source, sourceContent: SourceContent) => (
      <LanguageProvider>
        <JobActivityProvider>
          <SourcesViewerMode
            source={source}
            sourceContent={sourceContent}
            onRefresh={vi.fn()}
            listId="list-1"
          />
        </JobActivityProvider>
      </LanguageProvider>
    );

    const { rerender } = render(renderViewer(sourceA, contentA));

    // Wait for the sections fetch to resolve and the pager to render.
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      await Promise.resolve();
    });

    // Initially: section 1 of source A is shown.
    expect(screen.getByText(/Section 1 summary/)).toBeInTheDocument();

    // Navigate to section 3 (two Next clicks).
    await act(async () => {
      screen.getByLabelText("Next section").click();
    });
    await act(async () => {
      screen.getByLabelText("Next section").click();
    });
    expect(screen.getByText(/Section 3 summary/)).toBeInTheDocument();

    // Switch to source B. The `key={source.id}` remounts DigestAccordion,
    // resetting activeSectionIdx to 0. Sections for B arrive; the body
    // should show section 1 of source B, not section 3.
    rerender(renderViewer(sourceB, contentB));

    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByText(/Section 101 summary/)).toBeInTheDocument();
    // The previous-source section 3 ("Section 3 summary") must NOT
    // still be visible — that was the desync bug.
    expect(screen.queryByText(/^Section 3 summary$/)).not.toBeInTheDocument();
  });
});
