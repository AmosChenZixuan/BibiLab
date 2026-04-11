import { describe, expect, test } from "vitest";
import { readFileSync } from "fs";
import { resolve } from "path";

const css = readFileSync(resolve(__dirname, "../styles/app.css"), "utf-8");

describe("design tokens — Meta Store (phase 1: colors, spacing, shadow, radius, skeleton, glass)", () => {
  test("Plus Jakarta Sans font is imported", () => {
    expect(css).toContain("Plus Jakarta Sans");
    expect(css).toContain("fonts.googleapis.com");
  });

  test("--font-sans is defined with Plus Jakarta Sans", () => {
    expect(css).toMatch(/--font-sans:\s*["']?Plus Jakarta Sans/i);
  });

  test("--color-pink remapped to Meta Blue via alias", () => {
    const match = css.match(/--color-pink:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("var(--color-meta-blue)");
  });

  test("--color-blue remapped to Slate Gray (#5D6C7B)", () => {
    const match = css.match(/--color-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#5D6C7B");
  });

  test("--color-ink updated to #050505", () => {
    const match = css.match(/--color-ink:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#050505");
  });

  test("--color-muted updated to #65676B", () => {
    const match = css.match(/--color-muted:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#65676B");
  });

  test("--color-sky remapped to Baby Blue via alias", () => {
    const match = css.match(/--color-sky:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("var(--color-baby-blue)");
  });

  test("--color-border updated to #CED0D4", () => {
    const match = css.match(/--color-border:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#CED0D4");
  });

  test("--color-scrim updated to rgba(0,0,0,0.6)", () => {
    const match = css.match(/--color-scrim:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("rgba(0,0,0,0.6)");
  });

  test("--color-meta-blue is defined as #0064E0", () => {
    const match = css.match(/--color-meta-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#0064E0");
  });

  test("--color-meta-blue-hover is defined as #0143B5", () => {
    const match = css.match(/--color-meta-blue-hover:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#0143B5");
  });

  test("--color-meta-blue-pressed is defined as #004BB9", () => {
    const match = css.match(/--color-meta-blue-pressed:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#004BB9");
  });

  test("--color-meta-blue-light is defined as #47A5FA", () => {
    const match = css.match(/--color-meta-blue-light:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#47A5FA");
  });

  test("--color-charcoal is defined as #1C2B33", () => {
    const match = css.match(/--color-charcoal:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#1C2B33");
  });

  test("--color-baby-blue is defined as #E8F3FF", () => {
    const match = css.match(/--color-baby-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#E8F3FF");
  });

  test("--color-facebook-blue is defined as #1877F2", () => {
    const match = css.match(/--color-facebook-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#1877F2");
  });

  test("--color-soft-gray is defined as #F1F4F7", () => {
    const match = css.match(/--color-soft-gray:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#F1F4F7");
  });

  test("--color-warm-gray is defined as #F7F8FA", () => {
    const match = css.match(/--color-warm-gray:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#F7F8FA");
  });

  test("--color-web-wash is defined as #F0F2F5", () => {
    const match = css.match(/--color-web-wash:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#F0F2F5");
  });

  test("--color-near-black is defined as #1C1E21", () => {
    const match = css.match(/--color-near-black:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#1C1E21");
  });

  test("--color-success is defined as #31A24C", () => {
    const match = css.match(/--color-success:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#31A24C");
  });

  test("--color-error is defined as #E41E3F", () => {
    const match = css.match(/--color-error:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#E41E3F");
  });

  test("--color-warning is defined as #F7B928", () => {
    const match = css.match(/--color-warning:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#F7B928");
  });

  test("--color-disabled-bg is defined as #DEE3E9", () => {
    const match = css.match(/--color-disabled-bg:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#DEE3E9");
  });

  test("--color-disabled-text is defined as #8595A4", () => {
    const match = css.match(/--color-disabled-text:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#8595A4");
  });

  test("--color-secondary-hover is defined", () => {
    const match = css.match(/--color-secondary-hover:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("rgba(70,90,105,0.7)");
  });

  test("--color-secondary-border is defined", () => {
    const match = css.match(/--color-secondary-border:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("rgba(10,19,23,0.12)");
  });

  test("--spacing-button-x is defined as 22px", () => {
    const match = css.match(/--spacing-button-x:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("22px");
  });

  test("spacing scale --space-1 through --space-14 defined", () => {
    const expected: Record<string, string> = {
      "--space-1": "1px",
      "--space-2": "4px",
      "--space-3": "8px",
      "--space-4": "10px",
      "--space-5": "12px",
      "--space-6": "14px",
      "--space-7": "16px",
      "--space-8": "18px",
      "--space-9": "24px",
      "--space-10": "32px",
      "--space-11": "40px",
      "--space-12": "48px",
      "--space-13": "64px",
      "--space-14": "80px",
    };
    for (const [token, val] of Object.entries(expected)) {
      const match = css.match(new RegExp(`${token}:\\s*([^;]+)`));
      const found = match?.[1].trim();
      expect(found).toBe(val);
    }
  });

  test("--shadow-level-1 is defined", () => {
    expect(css).toContain("--shadow-level-1:");
    expect(css).toContain("rgba(0,0,0,0.1)");
  });

  test("--shadow-elevated is defined", () => {
    expect(css).toContain("--shadow-elevated:");
    expect(css).toContain("rgba(0,0,0,0.2)");
    expect(css).toContain("rgba(0,0,0,0.1)");
  });

  test("--radius-sm is defined as 8px", () => {
    const match = css.match(/--radius-sm:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("8px");
  });

  test("--radius-card is defined as 20px", () => {
    const match = css.match(/--radius-card:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("20px");
  });

  test("--radius-feature is defined as 24px", () => {
    const match = css.match(/--radius-feature:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("24px");
  });

  test("--radius-pill is defined as 100px", () => {
    const match = css.match(/--radius-pill:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("100px");
  });

  test("@keyframes glimmer is defined", () => {
    expect(css).toContain("@keyframes glimmer");
    expect(css).toContain("0%, 100% { opacity: 0.25; }");
    expect(css).toContain("50%       { opacity: 1.0; }");
  });

  test(".glimmer-placeholder is defined", () => {
    expect(css).toContain(".glimmer-placeholder");
    expect(css).toContain("background-color: #979A9F");
    expect(css).toContain("border-radius: 8px");
    expect(css).toContain("animation: glimmer 1000ms steps(1) infinite");
  });

  test(".glass is defined with backdrop-filter", () => {
    expect(css).toContain(".glass");
    expect(css).toContain("background-color: rgba(241, 244, 247, 0.8)");
    expect(css).toContain("backdrop-filter: blur(12px)");
  });

});
