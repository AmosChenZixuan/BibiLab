import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { DigestAccordion } from "@/components/lists/DigestAccordion";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { LanguageProvider } from "@/app/LanguageContext";
import type { SourceSection } from "@/lib/types";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const baseProps = {
  source: { id: "src-1" },
  summary: "Source-level summary (legacy, used for 1-section case)",
  keywords: ["alpha", "beta"],
  onRerun: vi.fn(),
  onRefresh: vi.fn(),
  facets: { seriesName: "Series A", sequenceNumber: 1, seasonNumber: null },
  onSaveFacets: vi.fn().mockResolvedValue(undefined),
  listId: "list-1",
};

function makeSections(n: number, startSec = 0): SourceSection[] {
  return Array.from({ length: n }, (_, i) => ({
    section_id: `sec-${i + 1}`,
    seq: i + 1,
    summary: `Section ${i + 1} summary text`,
    keywords: [`kw-${i + 1}a`, `kw-${i + 1}b`],
    timestamp_start: startSec + i * 600,
    timestamp_end: startSec + (i + 1) * 600,
  }));
}

function renderDigest(extra: Partial<React.ComponentProps<typeof DigestAccordion>> = {}) {
  return render(
    <LanguageProvider>
      <JobActivityProvider>
        <DigestAccordion {...baseProps} {...extra} />
      </JobActivityProvider>
    </LanguageProvider>,
  );
}

describe("DigestAccordion sections", () => {
  test("1-section case (sections undefined): renders the legacy markup, no pager", () => {
    renderDigest();
    // The 1-section path is byte-identical: source-level summary + chips,
    // no role="tablist".
    expect(screen.getByText(/Source-level summary/)).toBeInTheDocument();
    expect(screen.getAllByText(/^(alpha|beta)$/)).toHaveLength(2);
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });

  test("1-section case (sections has 1 item): still renders the legacy markup, no pager", () => {
    renderDigest({ sections: makeSections(1) });
    expect(screen.getByText(/Source-level summary/)).toBeInTheDocument();
    expect(screen.getAllByText(/^(alpha|beta)$/)).toHaveLength(2);
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });

  test("3-section case: all tabs visible, no arrows (page fits the visible window exactly)", () => {
    renderDigest({ sections: makeSections(3) });
    // Section summaries render via the active section (initially section 1).
    expect(screen.getByText(/Section 1 summary text/)).toBeInTheDocument();
    // Pager is rendered with the three tab buttons.
    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(3);
    // Arrows are hidden when the page fits exactly.
    expect(screen.queryByLabelText("Previous section")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Next section")).not.toBeInTheDocument();
  });

  test("5-section case: arrows visible, both enabled at the start", () => {
    renderDigest({ sections: makeSections(5) });
    const prev = screen.getByLabelText("Previous section");
    const next = screen.getByLabelText("Next section");
    expect(prev).toBeInTheDocument();
    expect(next).toBeInTheDocument();
    expect(prev).toBeDisabled();
    expect(next).toBeEnabled();
  });

  test("5-section case: clicking Next advances the active section and updates the body", async () => {
    const user = userEvent.setup();
    renderDigest({ sections: makeSections(5) });
    expect(screen.getByText(/Section 1 summary text/)).toBeInTheDocument();

    await user.click(screen.getByLabelText("Next section"));
    expect(screen.getByText(/Section 2 summary text/)).toBeInTheDocument();

    await user.click(screen.getByLabelText("Next section"));
    expect(screen.getByText(/Section 3 summary text/)).toBeInTheDocument();
  });

  test("5-section case: at the end, Next is disabled and Previous is enabled", async () => {
    const user = userEvent.setup();
    renderDigest({ sections: makeSections(5) });
    const prev = screen.getByLabelText("Previous section");
    const next = screen.getByLabelText("Next section");

    // Walk to the end (1 -> 2 -> 3 -> 4 -> 5 = 4 clicks).
    for (let i = 0; i < 4; i++) {
      await user.click(next);
    }

    expect(screen.getByText(/Section 5 summary text/)).toBeInTheDocument();
    expect(next).toBeDisabled();
    expect(prev).toBeEnabled();
  });

  test("5-section case: clicking a tab directly switches to that section", async () => {
    const user = userEvent.setup();
    renderDigest({ sections: makeSections(5) });
    const tabs = screen.getAllByRole("tab");
    // 4th tab = section 4 (tabs render in section order).
    await user.click(tabs[3]);
    expect(screen.getByText(/Section 4 summary text/)).toBeInTheDocument();
  });
});
