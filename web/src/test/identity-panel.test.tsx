import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import IdentityPanel from "@/components/layout/IdentityPanel";
import { LanguageProvider } from "@/app/LanguageContext";

afterEach(() => {
  cleanup();
});

describe("identity panel", () => {
  const defaultProps = {
    bilibiliCookie: "",
    onClose: vi.fn(),
    onLogin: vi.fn(),
    onLogout: vi.fn(),
  };

  test("renders bilibili platform with signed out state", () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="" />
      </LanguageProvider>,
    );

    expect(screen.getByText("Bilibili")).toBeInTheDocument();
    expect(screen.getByText("Not signed in")).toBeInTheDocument();
  });

  test("renders bilibili platform with signed in state when cookie is present", () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="some-cookie-value" />
      </LanguageProvider>,
    );

    expect(screen.getByText("Bilibili")).toBeInTheDocument();
    expect(screen.getByText("Signed in")).toBeInTheDocument();
  });

  test("shows sign in button when disconnected", () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="" />
      </LanguageProvider>,
    );

    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  test("shows sign out button when connected", () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="some-cookie-value" />
      </LanguageProvider>,
    );

    expect(screen.getByRole("button", { name: /sign out/i })).toBeInTheDocument();
  });

  test("calls onLogin when sign in button is clicked", async () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="" />
      </LanguageProvider>,
    );

    await userEvent.click(screen.getByRole("button", { name: /sign in/i }));
    expect(defaultProps.onLogin).toHaveBeenCalled();
  });

  test("calls onLogout when sign out button is clicked", async () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="some-cookie-value" />
      </LanguageProvider>,
    );

    await userEvent.click(screen.getByRole("button", { name: /sign out/i }));
    expect(defaultProps.onLogout).toHaveBeenCalled();
  });

  test("renders as menu with proper semantics", () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} />
      </LanguageProvider>,
    );

    expect(screen.getAllByRole("menu")).toHaveLength(1);
  });
});
