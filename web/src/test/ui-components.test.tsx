import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, test, vi } from "vitest";

afterEach(cleanup);
import { Button } from "@/components/ui/Button";
import { ContextMenu } from "@/components/ui/ContextMenu";
import { Modal } from "@/components/ui/Modal";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";

// ── Button ──────────────────────────────────────────────────────────────────
describe("Button", () => {
  test("renders primary variant", () => {
    render(<Button variant="primary">Save</Button>);
    const btn = screen.getByRole("button", { name: "Save" });
    expect(btn.className).toContain("bg-pink");
  });

  test("renders ghost variant", () => {
    render(<Button variant="ghost">Cancel</Button>);
    expect(screen.getByRole("button").className).toContain("border-blue");
  });

  test("renders danger variant", () => {
    render(<Button variant="danger">Delete</Button>);
    expect(screen.getByRole("button").className).toContain("bg-ink");
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
    expect(screen.getByText("OK").className).toContain("text-blue");
  });

  test("renders error status color", () => {
    render(<StatusChip status="error">Error</StatusChip>);
    expect(screen.getByText("Error").className).toContain("text-pink");
  });

  test("renders unavailable status color", () => {
    render(<StatusChip status="unavailable">Down</StatusChip>);
    expect(screen.getByText("Down").className).toContain("text-muted");
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
        trigger={({ toggle, triggerProps }) => (
          <button {...triggerProps} onClick={toggle} type="button">
            Menu
          </button>
        )}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Menu" }));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Delete" }).className).toContain("text-pink");

    await userEvent.click(document.body);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  test("keeps only one instance open at a time", async () => {
    function Example({ label }: { label: string }) {
      return (
        <ContextMenu
          items={[{ label: "Rename", onClick: () => {} }]}
          trigger={({ toggle, triggerProps }) => (
            <button {...triggerProps} onClick={toggle} type="button">
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

  test("trigger points at the open menu via aria-controls", async () => {
    render(
      <ContextMenu
        items={[{ label: "Rename", onClick: () => {} }]}
        trigger={({ toggle, triggerProps }) => (
          <button {...triggerProps} onClick={toggle} type="button">
            Menu
          </button>
        )}
      />,
    );

    const trigger = screen.getByRole("button", { name: "Menu" });
    expect(trigger).toHaveAttribute("aria-haspopup", "menu");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).not.toHaveAttribute("aria-controls");

    await userEvent.click(trigger);

    const menu = screen.getByRole("menu");
    expect(menu.id).toBeTruthy();
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(trigger).toHaveAttribute("aria-controls", menu.id);
  });

  test("menu id is stable across re-renders and unique per instance", async () => {
    function TwoMenus({ tick }: { tick: number }) {
      return (
        <>
          <span>tick {tick}</span>
          {["Menu A", "Menu B"].map((label) => (
            <ContextMenu
              items={[{ label: "Rename", onClick: () => {} }]}
              key={label}
              trigger={({ toggle, triggerProps }) => (
                <button {...triggerProps} onClick={toggle} type="button">
                  {label}
                </button>
              )}
            />
          ))}
        </>
      );
    }

    const { rerender } = render(<TwoMenus tick={1} />);

    const triggerA = screen.getByRole("button", { name: "Menu A" });
    await userEvent.click(triggerA);
    const idA = screen.getByRole("menu").id;
    expect(triggerA).toHaveAttribute("aria-controls", idA);

    rerender(<TwoMenus tick={2} />);
    expect(screen.getByRole("menu").id).toBe(idA);
    expect(screen.getByRole("button", { name: "Menu A" })).toHaveAttribute("aria-controls", idA);

    await userEvent.click(screen.getByRole("button", { name: "Menu B" }));
    const idB = screen.getByRole("menu").id;
    expect(idB).not.toBe(idA);
    expect(screen.getByRole("button", { name: "Menu B" })).toHaveAttribute("aria-controls", idB);
    expect(screen.getByRole("button", { name: "Menu A" })).not.toHaveAttribute("aria-controls");
  });
});
