import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import IdentityPanel from "@/components/layout/IdentityPanel";
import { LanguageProvider } from "@/app/LanguageContext";

afterEach(() => {
  cleanup();
});

describe("identity panel", () => {
  test("renders bilibili platform skeleton", () => {
    render(
      <LanguageProvider>
        <IdentityPanel onClose={() => {}} />
      </LanguageProvider>,
    );

    expect(screen.getByText("Bilibili")).toBeInTheDocument();
    expect(screen.getByText("Not signed in")).toBeInTheDocument();
  });

  test("renders as skeleton with menu semantics", () => {
    render(
      <LanguageProvider>
        <IdentityPanel onClose={() => {}} />
      </LanguageProvider>,
    );

    expect(screen.getAllByRole("menu")).toHaveLength(1);
  });
});
