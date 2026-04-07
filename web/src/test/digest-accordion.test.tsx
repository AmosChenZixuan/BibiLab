import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import { DigestAccordion } from "@/components/lists/DigestAccordion";
import { LanguageProvider } from "@/app/LanguageContext";

describe("DigestAccordion", () => {
  test("renders keywords as chips", () => {
    render(
      <LanguageProvider>
        <DigestAccordion
          source={{ id: "src-1" }}
          summary="foo bar"
          keywords={["alpha", "beta"]}
          onRerun={vi.fn()}
        />
      </LanguageProvider>,
    );

    // Accordion starts expanded by default
    const chips = screen.getAllByText(/^(alpha|beta)$/);
    expect(chips).toHaveLength(2);
  });
});
