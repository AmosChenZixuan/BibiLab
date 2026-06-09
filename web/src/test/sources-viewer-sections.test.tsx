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
    section_id: `sec-${offset + i + 1}`,
    seq: i + 1,
    summary: `Section ${offset + i + 1} summary`,
    keywords: [`kw-${offset + i + 1}`],
    timestamp_start: i * 600,
    timestamp_end: (i + 1) * 600,
  }));
}

describe("SourcesViewerMode sections — source-switch desync", () => {
  test("targetSection.sectionId jumps directly to the cited section", async () => {
    // The sectionId match path is the precise match: the chat citation
    // carries the section's id directly so it lands on the cited
    // section even when the chunk-anchored timestamp would fall at a
    // section boundary or outside any section's range. 3 sections
    // cover [0,600], [600,1200], [1200,1800]. targetSection with
    // sectionId="sec-3" and timestampStart=300 must land on section 3
    // (the sectionId branch), NOT section 1 (where 300 falls) — so
    // the assertion is load-bearing: if the sectionId and timestamp
    // branches were ever swapped, the test would fail.
    const contentA = makeContent("src-A", "Source A", "Source A summary");
    const sectionsA = makeSections(3, 0);

    vi.spyOn(api, "getSource").mockResolvedValue(contentA);
    vi.spyOn(api, "getSourceSections").mockResolvedValue(sectionsA);
    vi.spyOn(api, "listJobs").mockResolvedValue([]);

    const sourceA = makeSource("src-A", "Source A");
    render(
      <LanguageProvider>
        <JobActivityProvider>
          <SourcesViewerMode
            source={sourceA}
            sourceContent={contentA}
            onRefresh={vi.fn()}
            listId="list-1"
            targetSection={{ sectionId: "sec-3", timestampStart: 300 }}
          />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    // Let the sections fetch resolve.
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      await Promise.resolve();
    });

    // Section 3 is active: its summary is in the body, the 3rd tab is
    // aria-selected, and the siblings are not. (Section 1, where 300
    // would land on the timestamp branch, is explicitly asserted NOT
    // to be active.)
    expect(screen.getByText(/Section 3 summary/)).toBeInTheDocument();
    expect(screen.queryByText(/^Section 1 summary$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Section 2 summary$/)).not.toBeInTheDocument();
    const tabs = screen.getAllByRole("tab");
    expect(tabs[2]).toHaveAttribute("aria-selected", "true");
    expect(tabs[0]).toHaveAttribute("aria-selected", "false");
    expect(tabs[1]).toHaveAttribute("aria-selected", "false");
  });

  test("targetSection selects the containing section tab on open", async () => {
    // 3 sections covering [0,600], [600,1200], [1200,1800]. targetSection
    // with timestampStart=700 should land in section 2 — the only section
    // whose [start,end] range contains 700 unambiguously. The handoff
    // pins the matcher to timestamp (SourceSection has no section_id at
    // runtime), so the test exercises the timestamp path directly.
    const contentA = makeContent("src-A", "Source A", "Source A summary");
    const sectionsA = makeSections(3, 0);

    vi.spyOn(api, "getSource").mockResolvedValue(contentA);
    vi.spyOn(api, "getSourceSections").mockResolvedValue(sectionsA);
    vi.spyOn(api, "listJobs").mockResolvedValue([]);

    const sourceA = makeSource("src-A", "Source A");
    render(
      <LanguageProvider>
        <JobActivityProvider>
          <SourcesViewerMode
            source={sourceA}
            sourceContent={contentA}
            onRefresh={vi.fn()}
            listId="list-1"
            targetSection={{ timestampStart: 700 }}
          />
        </JobActivityProvider>
      </LanguageProvider>,
    );

    // Let the sections fetch resolve.
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      await Promise.resolve();
    });

    // Section 2 is active: its summary is in the body, the 2nd tab is
    // aria-selected, and the siblings are not.
    expect(screen.getByText(/Section 2 summary/)).toBeInTheDocument();
    expect(screen.queryByText(/^Section 1 summary$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Section 3 summary$/)).not.toBeInTheDocument();
    const tabs = screen.getAllByRole("tab");
    expect(tabs[1]).toHaveAttribute("aria-selected", "true");
    expect(tabs[0]).toHaveAttribute("aria-selected", "false");
    expect(tabs[2]).toHaveAttribute("aria-selected", "false");
  });

  test("re-jump to the same source re-syncs the active section", async () => {
    // Sections: [0,600], [600,1200], [1200,1800]. The first citation
    // lands in section 3 (timestamp 1500). The second citation lands in
    // section 1 (timestamp 300). The DigestAccordion `key={source.id}`
    // does NOT change between the two jumps (same source), so the
    // effect on `initialActiveIdx` — not the mount-time state seed —
    // is what drives the second switch.
    const contentA = makeContent("src-A", "Source A", "Source A summary");
    const sectionsA = makeSections(3, 0);

    vi.spyOn(api, "getSource").mockResolvedValue(contentA);
    vi.spyOn(api, "getSourceSections").mockResolvedValue(sectionsA);
    vi.spyOn(api, "listJobs").mockResolvedValue([]);

    const sourceA = makeSource("src-A", "Source A");

    const renderViewer = (target: number) => (
      <LanguageProvider>
        <JobActivityProvider>
          <SourcesViewerMode
            source={sourceA}
            sourceContent={contentA}
            onRefresh={vi.fn()}
            listId="list-1"
            targetSection={{ timestampStart: target }}
          />
        </JobActivityProvider>
      </LanguageProvider>
    );

    // Lower interior point of section 3 — section 3 covers [1200,1800],
    // 1500 is unambiguously inside. The tab strip is centered on the
    // initial section; we only assert which tab is aria-selected.
    const { rerender } = render(renderViewer(1500));

    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByText(/Section 3 summary/)).toBeInTheDocument();
    const tabsAfter1500 = screen.getAllByRole("tab");
    expect(tabsAfter1500[2]).toHaveAttribute("aria-selected", "true");

    // Now re-jump to a timestamp in section 1 (e.g. 300). The
    // DigestAccordion `key={source.id}` does not change (same source),
    // so the effect on `initialActiveIdx` must drive the tab switch.
    rerender(renderViewer(300));

    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByText(/Section 1 summary/)).toBeInTheDocument();
    const tabsAfter300 = screen.getAllByRole("tab");
    expect(tabsAfter300[0]).toHaveAttribute("aria-selected", "true");
    expect(tabsAfter300[1]).toHaveAttribute("aria-selected", "false");
    expect(tabsAfter300[2]).toHaveAttribute("aria-selected", "false");
  });

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
