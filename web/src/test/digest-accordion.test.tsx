import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { DigestAccordion } from "@/components/lists/DigestAccordion";
import { LanguageProvider } from "@/app/LanguageContext";

describe("DigestAccordion", () => {
  test("renders keywords as chips", () => {
    render(
      <LanguageProvider>
        <DigestAccordion summary="foo bar" keywords={["alpha", "beta"]} />
      </LanguageProvider>,
    );

    // Click the accordion header to expand
    const header = screen.getByRole("button");
    fireEvent.click(header);

    // Assert two chip elements are present
    const chips = screen.getAllByText(/^(alpha|beta)$/);
    expect(chips).toHaveLength(2);

    // Assert neither chip has an onClick handler
    for (const chip of chips) {
      expect(chip).not.toHaveAttribute("onClick");
      expect(chip).not.toHaveAttribute("onclick");
    }
  });
});
