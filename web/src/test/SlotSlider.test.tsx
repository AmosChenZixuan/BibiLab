import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, test } from "vitest";

import { SlotSlider } from "@/components/ui/SlotSlider";
import type { SlotOption } from "@/components/ui/SlotSlider";

afterEach(() => {
  cleanup();
});

const OPTIONS: SlotOption<number>[] = [
  { value: 1, label: "A" },
  { value: 2, label: "B" },
  { value: 3, label: "C" },
];

function Harness({ initial = 1 }: { initial?: number }) {
  const [value, setValue] = useState(initial);
  return <SlotSlider ariaLabel="Test" options={OPTIONS} value={value} onChange={setValue} />;
}

describe("SlotSlider", () => {
  test("arrow key selects the next slot and moves DOM focus to it", () => {
    render(<Harness />);
    const group = screen.getByRole("radiogroup", { name: "Test" });
    const a = group.querySelector('[role="radio"][aria-label="A"]') as HTMLElement;
    const b = group.querySelector('[role="radio"][aria-label="B"]') as HTMLElement;

    a.focus();
    fireEvent.keyDown(a, { key: "ArrowRight" });

    expect(b).toHaveAttribute("aria-checked", "true");
    // Focus must follow selection — otherwise the focus ring is stranded on a
    // now-unselected (tabIndex=-1) button.
    expect(b).toHaveFocus();
    expect(b).toHaveAttribute("tabIndex", "0");
    expect(a).toHaveAttribute("tabIndex", "-1");
  });

  test("End selects and focuses the last slot, Home the first", () => {
    render(<Harness />);
    const group = screen.getByRole("radiogroup", { name: "Test" });
    const a = group.querySelector('[role="radio"][aria-label="A"]') as HTMLElement;
    const c = group.querySelector('[role="radio"][aria-label="C"]') as HTMLElement;

    a.focus();
    fireEvent.keyDown(a, { key: "End" });
    expect(c).toHaveAttribute("aria-checked", "true");
    expect(c).toHaveFocus();

    fireEvent.keyDown(c, { key: "Home" });
    expect(a).toHaveAttribute("aria-checked", "true");
    expect(a).toHaveFocus();
  });
});
