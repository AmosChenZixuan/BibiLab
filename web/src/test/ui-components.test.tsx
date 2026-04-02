import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, test, vi } from "vitest";

afterEach(cleanup);
import { Button } from "../components/ui/Button";
import { FormField } from "../components/ui/FormField";
import { Panel } from "../components/ui/Panel";
import { StatusChip } from "../components/ui/StatusChip";

// ── Button ──────────────────────────────────────────────────────────────────
describe("Button", () => {
  test("renders primary variant", () => {
    render(<Button variant="primary">Save</Button>);
    const btn = screen.getByRole("button", { name: "Save" });
    expect(btn.className).toContain("from-pink");
    expect(btn.className).toContain("to-blue");
  });

  test("renders ghost variant", () => {
    render(<Button variant="ghost">Cancel</Button>);
    expect(screen.getByRole("button").className).toContain("border-blue");
  });

  test("renders danger variant", () => {
    render(<Button variant="danger">Delete</Button>);
    expect(screen.getByRole("button").className).toContain("bg-danger");
  });

  test("forwards className prop", () => {
    render(<Button variant="secondary" className="mt-4">Go</Button>);
    expect(screen.getByRole("button").className).toContain("mt-4");
  });

  test("forwards disabled prop", () => {
    render(<Button variant="primary" disabled>X</Button>);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("applies sm size classes", () => {
    render(<Button size="sm">Small</Button>);
    expect(screen.getByRole("button").className).toContain("text-sm");
  });
});

// ── FormField ────────────────────────────────────────────────────────────────
describe("FormField", () => {
  test("renders label", () => {
    render(<FormField label="Email"><input /></FormField>);
    expect(screen.getByText("Email")).toBeInTheDocument();
  });

  test("renders hint when provided", () => {
    render(<FormField label="Name" hint="Required"><input /></FormField>);
    expect(screen.getByText("Required")).toBeInTheDocument();
  });

  test("does not render hint when omitted", () => {
    render(<FormField label="Name"><input /></FormField>);
    expect(screen.queryByText("Required")).not.toBeInTheDocument();
  });

  test("forwards className", () => {
    const { container } = render(<FormField label="X" className="mt-2"><input /></FormField>);
    expect(container.firstChild as HTMLElement).toHaveClass("mt-2");
  });
});

// ── Panel ────────────────────────────────────────────────────────────────────
describe("Panel", () => {
  test("renders app variant with surface bg", () => {
    const { container } = render(<Panel variant="app"><p>content</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("bg-surface");
  });

  test("renders workspace variant", () => {
    const { container } = render(<Panel variant="workspace"><p>content</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("bg-white");
  });

  test("defaults to app variant", () => {
    const { container } = render(<Panel><p>x</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("bg-surface");
  });

  test("forwards className", () => {
    const { container } = render(<Panel className="p-8"><p>x</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("p-8");
  });
});

// ── StatusChip ───────────────────────────────────────────────────────────────
describe("StatusChip", () => {
  test("renders ok status color", () => {
    render(<StatusChip status="ok">OK</StatusChip>);
    expect(screen.getByText("OK").className).toContain("text-success");
  });

  test("renders error status color", () => {
    render(<StatusChip status="error">Error</StatusChip>);
    expect(screen.getByText("Error").className).toContain("text-danger");
  });

  test("renders unavailable status color", () => {
    render(<StatusChip status="unavailable">Down</StatusChip>);
    expect(screen.getByText("Down").className).toContain("text-warn");
  });

  test("renders neutral status by default", () => {
    render(<StatusChip>Unknown</StatusChip>);
    expect(screen.getByText("Unknown").className).toContain("text-blue");
  });

  test("forwards className", () => {
    render(<StatusChip status="ok" className="ml-2">OK</StatusChip>);
    expect(screen.getByText("OK").className).toContain("ml-2");
  });
});
