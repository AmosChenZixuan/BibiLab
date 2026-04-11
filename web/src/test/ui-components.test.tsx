import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, test, vi } from "vitest";

afterEach(cleanup);
import { Button } from "@/components/ui/Button";
import { ContextMenu } from "@/components/ui/ContextMenu";
import { Modal } from "@/components/ui/Modal";
import { FormField } from "@/components/ui/FormField";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";

// ── Button ──────────────────────────────────────────────────────────────────
describe("Button", () => {
  test("renders primary variant", () => {
    render(<Button variant="primary">Save</Button>);
    const btn = screen.getByRole("button", { name: "Save" });
    expect(btn.className).toContain("bg-meta-blue");
  });

  test("renders ghost variant", () => {
    render(<Button variant="ghost">Cancel</Button>);
    expect(screen.getByRole("button").className).toContain("text-meta-blue");
  });

  test("renders danger variant", () => {
    render(<Button variant="danger">Delete</Button>);
    expect(screen.getByRole("button").className).toContain("bg-error");
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
    expect(screen.getByRole("button").className).toContain("px-3");
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
  test("renders app variant with translucent white bg", () => {
    const { container } = render(<Panel variant="app"><p>content</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("bg-white/80");
  });

  test("renders workspace variant", () => {
    const { container } = render(<Panel variant="workspace"><p>content</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("bg-white/76");
  });

  test("defaults to app variant", () => {
    const { container } = render(<Panel><p>x</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("bg-white/80");
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
    expect(screen.getByText("OK").className).toContain("bg-success");
  });

  test("renders error status color", () => {
    render(<StatusChip status="error">Error</StatusChip>);
    expect(screen.getByText("Error").className).toContain("bg-error");
  });

  test("renders unavailable status color", () => {
    render(<StatusChip status="unavailable">Down</StatusChip>);
    expect(screen.getByText("Down").className).toContain("text-charcoal");
  });

  test("renders neutral status by default", () => {
    render(<StatusChip>Unknown</StatusChip>);
    expect(screen.getByText("Unknown").className).toContain("bg-meta-blue");
  });

  test("forwards className", () => {
    render(<StatusChip status="ok" className="ml-2">OK</StatusChip>);
    expect(screen.getByText("OK").className).toContain("ml-2");
  });
});

// ── Modal ────────────────────────────────────────────────────────────────────
describe("Modal", () => {
  test("renders when open and closes on escape or backdrop click", async () => {
    const onClose = vi.fn();
    render(
      <Modal open onClose={onClose} title="Delete list">
        <p>Body</p>
      </Modal>,
    );

    expect(screen.getByRole("dialog", { name: "Delete list" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /close dialog/i })).not.toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByTestId("modal-backdrop"));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  test("does not close when interaction starts inside the modal and ends on the backdrop", () => {
    const onClose = vi.fn();
    render(
      <Modal open onClose={onClose} title="Rename list">
        <input aria-label="List name" defaultValue="Systems" />
      </Modal>,
    );

    fireEvent.mouseDown(screen.getByRole("dialog", { name: "Rename list" }));
    fireEvent.click(screen.getByTestId("modal-backdrop"));

    expect(onClose).not.toHaveBeenCalled();
  });

  test("does not render when closed", () => {
    render(
      <Modal open={false} onClose={() => {}} title="Closed">
        <p>Hidden</p>
      </Modal>,
    );

    expect(screen.queryByRole("dialog", { name: "Closed" })).not.toBeInTheDocument();
  });
});

// ── ContextMenu ──────────────────────────────────────────────────────────────
describe("ContextMenu", () => {
  test("opens from trigger, closes on outside click, and styles danger items", async () => {
    const onDelete = vi.fn();

    render(
      <ContextMenu
        items={[
          { label: "Rename", onClick: () => {} },
          { label: "Delete", onClick: onDelete, variant: "danger" },
        ]}
        trigger={({ open, toggle, triggerRef }) => (
          <button
            ref={triggerRef}
            aria-expanded={open}
            onClick={toggle}
            type="button"
          >
            Menu
          </button>
        )}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Menu" }));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Delete" }).className).toContain("text-error");

    await userEvent.click(document.body);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  test("keeps only one instance open at a time", async () => {
    function Example({ label }: { label: string }) {
      return (
        <ContextMenu
          items={[{ label: "Rename", onClick: () => {} }]}
          trigger={({ open, toggle, triggerRef }) => (
            <button ref={triggerRef} aria-expanded={open} onClick={toggle} type="button">
              {label}
            </button>
          )}
        />
      );
    }

    render(
      <>
        <Example label="Menu A" />
        <Example label="Menu B" />
      </>,
    );

    await userEvent.click(screen.getByRole("button", { name: "Menu A" }));
    expect(screen.getAllByRole("menu")).toHaveLength(1);

    await userEvent.click(screen.getByRole("button", { name: "Menu B" }));
    expect(screen.getAllByRole("menu")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Menu A" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: "Menu B" })).toHaveAttribute("aria-expanded", "true");
  });
});
