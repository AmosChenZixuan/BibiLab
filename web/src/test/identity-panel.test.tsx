import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import IdentityPanel from "../components/layout/IdentityPanel";

afterEach(() => {
  cleanup();
});

describe("identity panel", () => {
  test("renders bilibili platform skeleton", () => {
    render(<IdentityPanel onClose={() => {}} />);

    expect(screen.getByText("Bilibili")).toBeInTheDocument();
    expect(screen.getByText("Not signed in")).toBeInTheDocument();
  });

  test("renders as skeleton with menu semantics", () => {
    render(<IdentityPanel onClose={() => {}} />);

    expect(screen.getByRole("menu")).toBeInTheDocument();
  });
});
