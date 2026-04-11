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

  test("--color-pink is defined as #FF66BF", () => {
    const match = css.match(/--color-pink:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#FF66BF");
  });

  test("--color-pink-hover is defined as #FF85C8", () => {
    const match = css.match(/--color-pink-hover:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#FF85C8");
  });

  test("--color-pink-pressed is defined as #E050A8", () => {
    const match = css.match(/--color-pink-pressed:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#E050A8");
  });

  test("--color-pink-light is defined as #FFD6ED", () => {
    const match = css.match(/--color-pink-light:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#FFD6ED");
  });

  test("--color-sky-blue is defined as #87CEEB", () => {
    const match = css.match(/--color-sky-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#87CEEB");
  });

  test("--color-sky-blue-hover is defined as #6BB8DB", () => {
    const match = css.match(/--color-sky-blue-hover:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#6BB8DB");
  });

  test("--color-sky-blue-pressed is defined as #4AA8CB", () => {
    const match = css.match(/--color-sky-blue-pressed:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#4AA8CB");
  });

  test("--color-sky-blue-light is defined as #D6EDFA", () => {
    const match = css.match(/--color-sky-blue-light:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#D6EDFA");
  });

  test("--color-primary-text is defined as #050505", () => {
    const match = css.match(/--color-primary-text:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#050505");
  });

  test("--color-charcoal is defined as #1C2B33", () => {
    const match = css.match(/--color-charcoal:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#1C2B33");
  });

  test("--color-secondary-text is defined as #65676B", () => {
    const match = css.match(/--color-secondary-text:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#65676B");
  });

  test("--color-slate-gray is defined as #5D6C7B", () => {
    const match = css.match(/--color-slate-gray:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#5D6C7B");
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

  test("--color-facebook-blue is defined as #1877F2", () => {
    const match = css.match(/--color-facebook-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#1877F2");
  });

  test("--color-rayban-red is defined as #D6311F", () => {
    const match = css.match(/--color-rayban-red:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#D6311F");
  });

  test("--color-oculus-purple is defined as #A121CE", () => {
    const match = css.match(/--color-oculus-purple:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#A121CE");
  });

  test("--color-portal-blue is defined as #1B365D", () => {
    const match = css.match(/--color-portal-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#1B365D");
  });

  test("--color-portal-hero-blue is defined as #C8E4E8", () => {
    const match = css.match(/--color-portal-hero-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#C8E4E8");
  });

  test("--color-portal-light-blue is defined as #ADD4E0", () => {
    const match = css.match(/--color-portal-light-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#ADD4E0");
  });

  test("--color-white is defined as #FFFFFF", () => {
    const match = css.match(/--color-white:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#FFFFFF");
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

  test("--color-linen is defined as #F2F0E6", () => {
    const match = css.match(/--color-linen:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#F2F0E6");
  });

  test("--color-baby-blue is defined as #E8F3FF", () => {
    const match = css.match(/--color-baby-blue:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#E8F3FF");
  });

  test("--color-near-black is defined as #1C1E21", () => {
    const match = css.match(/--color-near-black:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#1C1E21");
  });

  test("--color-oculus-light is defined as #181A1B", () => {
    const match = css.match(/--color-oculus-light:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#181A1B");
  });

  test("--color-overlay is defined as rgba(0,0,0,0.6)", () => {
    const match = css.match(/--color-overlay:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("rgba(0,0,0,0.6)");
  });

  test("--color-icon-secondary is defined as #465A69", () => {
    const match = css.match(/--color-icon-secondary:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#465A69");
  });

  test("--color-section-header is defined as #4B4C4F", () => {
    const match = css.match(/--color-section-header:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#4B4C4F");
  });

  test("--color-button-text-gray is defined as #444950", () => {
    const match = css.match(/--color-button-text-gray:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#444950");
  });

  test("--color-disabled-text is defined as #BCC0C4", () => {
    const match = css.match(/--color-disabled-text:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#BCC0C4");
  });

  test("--color-cta-disabled-text is defined as #8595A4", () => {
    const match = css.match(/--color-cta-disabled-text:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#8595A4");
  });

  test("--color-divider is defined as #CED0D4", () => {
    const match = css.match(/--color-divider:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#CED0D4");
  });

  test("--color-divider-gray is defined as #DEE3E9", () => {
    const match = css.match(/--color-divider-gray:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#DEE3E9");
  });

  test("--color-success is defined as #31A24C", () => {
    const match = css.match(/--color-success:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#31A24C");
  });

  test("--color-store-success is defined as #007D1E", () => {
    const match = css.match(/--color-store-success:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#007D1E");
  });

  test("--color-error is defined as #E41E3F", () => {
    const match = css.match(/--color-error:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#E41E3F");
  });

  test("--color-store-error is defined as #C80A28", () => {
    const match = css.match(/--color-store-error:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#C80A28");
  });

  test("--color-warning is defined as #F7B928", () => {
    const match = css.match(/--color-warning:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#F7B928");
  });

  test("--color-positive-bg is defined", () => {
    const match = css.match(/--color-positive-bg:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("rgba(36, 228, 0, 0.15)");
  });

  test("--color-error-bg is defined", () => {
    const match = css.match(/--color-error-bg:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("rgba(255, 123, 145, 0.15)");
  });

  test("--color-warning-bg is defined", () => {
    const match = css.match(/--color-warning-bg:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("rgba(255, 226, 0, 0.15)");
  });

  test("--color-info-bg is defined", () => {
    const match = css.match(/--color-info-bg:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("rgba(0, 145, 255, 0.15)");
  });

  test("--color-cherry is defined as #F3425F", () => {
    const match = css.match(/--color-cherry:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#F3425F");
  });

  test("--color-grape is defined as #9360F7", () => {
    const match = css.match(/--color-grape:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#9360F7");
  });

  test("--color-lime is defined as #45BD62", () => {
    const match = css.match(/--color-lime:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#45BD62");
  });

  test("--color-seafoam is defined as #54C7EC", () => {
    const match = css.match(/--color-seafoam:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#54C7EC");
  });

  test("--color-teal is defined as #2ABBA7", () => {
    const match = css.match(/--color-teal:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#2ABBA7");
  });

  test("--color-tomato is defined as #FB724B", () => {
    const match = css.match(/--color-tomato:\s*([^;]+)/);
    expect(match?.[1].trim()).toBe("#FB724B");
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

  test("--shadow-level-2 is defined", () => {
    expect(css).toContain("--shadow-level-2:");
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
