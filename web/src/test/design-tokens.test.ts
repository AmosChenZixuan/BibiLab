import { describe, expect, test } from "vitest";
import { readFileSync } from "fs";
import { resolve } from "path";

const css = readFileSync(resolve(__dirname, "../styles/app.css"), "utf-8");

describe("design tokens — Refined Soft", () => {
  test("Plus Jakarta Sans font is imported", () => {
    expect(css).toContain("Plus+Jakarta+Sans");
    expect(css).toContain("fonts.googleapis.com");
  });

  test("--font-sans is defined with Plus Jakarta Sans", () => {
    expect(css).toMatch(/--font-sans:\s*["']?Plus Jakarta Sans/i);
  });

  test("--color-ink is updated to slate-700 (#334155)", () => {
    expect(css).toContain("#334155");
  });

  test("--color-muted is updated to slate-500 (#64748b)", () => {
    expect(css).toContain("#64748b");
  });

  test("--color-border uses 18% opacity slate-500", () => {
    expect(css).toMatch(/--color-border:\s*rgba\(\s*100,\s*116,\s*139,\s*0\.18\s*\)/);
  });

  test("--font-serif is removed or commented out", () => {
    const serifLine = css.split("\n").find((l) => l.includes("--font-serif"));
    // Either removed entirely, or commented out with //
    const isRemoved =
      serifLine === undefined ||
      serifLine.trim().startsWith("//") ||
      serifLine.trim().startsWith("/*");
    expect(isRemoved).toBe(true);
  });
});
