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
    bilibiliUsername: "",
    bilibiliAvatarUrl: "",
    onClose: vi.fn(),
    onLogin: vi.fn(),
    onLogout: vi.fn(),
  };

  test("renders list-row layout with bilibili row showing username and avatar when signed in", () => {
    render(
      <LanguageProvider>
        <IdentityPanel
          {...defaultProps}
          bilibiliCookie="SESSDATA=abc"
          bilibiliUsername="test_user"
          bilibiliAvatarUrl="https://i0.hdslb.com/bfs/face/abc.jpg"
        />
      </LanguageProvider>,
    );

    expect(screen.getByText("Bilibili")).toBeInTheDocument();
    expect(screen.getByText("test_user")).toBeInTheDocument();
    const avatar = screen.getByRole("img");
    expect(avatar).toHaveAttribute(
      "src",
      "/api/proxy/cover?url=https%3A%2F%2Fi0.hdslb.com%2Fbfs%2Fface%2Fabc.jpg",
    );
    expect(screen.getByLabelText("Sign out")).toBeInTheDocument();
  });

  test("shows signed-out state when cookie is absent", () => {
    render(
      <LanguageProvider>
        <IdentityPanel
          {...defaultProps}
          bilibiliCookie=""
          bilibiliUsername=""
          bilibiliAvatarUrl=""
        />
      </LanguageProvider>,
    );

    expect(screen.getByText("Not signed in")).toBeInTheDocument();
    expect(screen.queryByText("test_user")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Sign in")).toBeInTheDocument();
  });

  test("list-row layout row exists for platform", () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="cookie" bilibiliUsername="u" bilibiliAvatarUrl="" />
      </LanguageProvider>,
    );

    const row = screen.getByTestId("bilibili-row");
    expect(row).toBeInTheDocument();
  });

  test("calls onLogin when sign in button is clicked", async () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="" />
      </LanguageProvider>,
    );

    await userEvent.click(screen.getByLabelText("Sign in"));
    expect(defaultProps.onLogin).toHaveBeenCalled();
  });

  test("calls onLogout when sign out button is clicked", async () => {
    render(
      <LanguageProvider>
        <IdentityPanel {...defaultProps} bilibiliCookie="some-cookie-value" />
      </LanguageProvider>,
    );

    await userEvent.click(screen.getByLabelText("Sign out"));
    expect(defaultProps.onLogout).toHaveBeenCalled();
  });

  test("shows navbar.signedIn text when cookie is present but username is empty", () => {
    render(
      <LanguageProvider>
        <IdentityPanel
          {...defaultProps}
          bilibiliCookie="SESSDATA=abc"
          bilibiliUsername=""
          bilibiliAvatarUrl=""
        />
      </LanguageProvider>,
    );

    expect(screen.getByText("Signed in")).toBeInTheDocument();
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
