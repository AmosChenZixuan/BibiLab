import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { DigestAccordion } from "@/components/lists/DigestAccordion";
import { LanguageProvider } from "@/app/LanguageContext";

afterEach(() => {
  cleanup();
});

const baseProps = {
  source: { id: "src-1" },
  summary: "foo bar",
  keywords: ["alpha", "beta"],
  onRerun: vi.fn(),
  facets: { seriesName: "罗翔说刑法", sequenceNumber: 8, seasonNumber: null },
  onSaveFacets: vi.fn().mockResolvedValue(undefined),
};

describe("DigestAccordion", () => {
  test("renders keywords as chips", () => {
    render(
      <LanguageProvider>
        <DigestAccordion {...baseProps} />
      </LanguageProvider>,
    );
    expect(screen.getAllByText(/^(alpha|beta)$/)).toHaveLength(2);
  });

  test("shows facet strip and an Edit metadata menu item", () => {
    render(
      <LanguageProvider>
        <DigestAccordion {...baseProps} />
      </LanguageProvider>,
    );
    expect(screen.getByText(/罗翔说刑法/)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Digest options"));
    expect(screen.getByText("Edit metadata")).toBeInTheDocument();
  });
});
