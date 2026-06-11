import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { DigestAccordion } from "@/components/lists/DigestAccordion";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { LanguageProvider } from "@/app/LanguageContext";
import { makeSections } from "@/test/utils";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const baseProps = {
  source: { id: "src-1" },
  onRerun: vi.fn(),
  onRefresh: vi.fn(),
  facets: { seriesName: "Series A", sequenceNumber: 1, seasonNumber: null },
  onSaveFacets: vi.fn().mockResolvedValue(undefined),
  listId: "list-1",
};

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
  test("sections not yet loaded (undefined): no digest body, no pager", () => {
    renderDigest();
    // Sections are the sole digest store; until they arrive there is no
    // summary/keywords to show and no pager.
    expect(screen.queryByText(/summary text/)).not.toBeInTheDocument();
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });

  test("1-section case: renders section 0's summary + keywords, no pager", () => {
    renderDigest({ sections: makeSections(1, { keywordsPerSection: 2, summarySuffix: " text" }) });
    expect(screen.getByText(/Section 1 summary text/)).toBeInTheDocument();
    expect(screen.getAllByText(/^(kw-1a|kw-1b)$/)).toHaveLength(2);
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });

  test("3-section case: all tabs visible, no arrows (page fits the visible window exactly)", () => {
    renderDigest({ sections: makeSections(3, { keywordsPerSection: 2, summarySuffix: " text" }) });
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
    renderDigest({ sections: makeSections(5, { keywordsPerSection: 2, summarySuffix: " text" }) });
    const prev = screen.getByLabelText("Previous section");
    const next = screen.getByLabelText("Next section");
    expect(prev).toBeInTheDocument();
    expect(next).toBeInTheDocument();
    expect(prev).toBeDisabled();
    expect(next).toBeEnabled();
  });

  test("5-section case: clicking Next advances the active section and updates the body", async () => {
    const user = userEvent.setup();
    renderDigest({ sections: makeSections(5, { keywordsPerSection: 2, summarySuffix: " text" }) });
    expect(screen.getByText(/Section 1 summary text/)).toBeInTheDocument();

    await user.click(screen.getByLabelText("Next section"));
    expect(screen.getByText(/Section 2 summary text/)).toBeInTheDocument();

    await user.click(screen.getByLabelText("Next section"));
    expect(screen.getByText(/Section 3 summary text/)).toBeInTheDocument();
  });

  test("5-section case: at the end, Next is disabled and Previous is enabled", async () => {
    const user = userEvent.setup();
    renderDigest({ sections: makeSections(5, { keywordsPerSection: 2, summarySuffix: " text" }) });
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
    renderDigest({ sections: makeSections(5, { keywordsPerSection: 2, summarySuffix: " text" }) });
    const tabs = screen.getAllByRole("tab");
    // 4th tab = section 4 (tabs render in section order).
    await user.click(tabs[3]);
    expect(screen.getByText(/Section 4 summary text/)).toBeInTheDocument();
  });
});
