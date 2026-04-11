import { describe, expect, test } from "vitest";
import { readFileSync, readdirSync } from "fs";
import { resolve } from "path";

const css = readFileSync(resolve(__dirname, "../styles/app.css"), "utf-8");

const COLOR_PREFIXES = ["text-", "bg-", "border-", "divide-"];

const TAILWIND_BUILT_IN_COLORS = new Set([
  "pink", "sky", "blue", "indigo", "violet", "purple", "fuchsia",
  "red", "orange", "amber", "yellow", "lime", "green", "emerald",
  "teal", "cyan", "slate", "gray", "neutral", "stone",
  "zinc", "black", "white", "transparent", "current", "inherit", "auto",
  "error", "success", "warning", "info",
  "rose", "fuchsia", "indigo", "cyan",
  "primary", "secondary", "accent", "muted", "foreground", "background",
  "destructive", "muted-foreground", "card-foreground", "popover-foreground",
  "primary-foreground", "secondary-foreground", "destructive-foreground",
  "accent-foreground", "chart", "sidebar", "sidebar-foreground",
  "sidebar-primary", "sidebar-primary-foreground", "sidebar-accent",
  "sidebar-accent-foreground", "sidebar-border", "sidebar-ring",
]);

function extractTokensFromTheme(cssContent: string): Set<string> {
  const tokens = new Set<string>();
  const themeMatch = cssContent.match(/@theme\s*\{([^}]+)\}/s);
  if (!themeMatch) return tokens;

  const themeBody = themeMatch[1];
  const tokenMatches = themeBody.matchAll(/--[\w-]+:\s*([^;]+)/g);
  for (const match of tokenMatches) {
    const fullToken = match[0].split(":")[0].replace("--", "");
    const normalized = fullToken.replace(/^(color|spacing|radius|shadow|z)-/, "");
    tokens.add(normalized);
  }

  const utilityMatches = cssContent.matchAll(/@utility\s+([\w-]+)/g);
  for (const match of utilityMatches) {
    const utilityName = match[1];
    const normalized = utilityName.replace(/^(text|bg|border|divide)-/, "");
    tokens.add(normalized);
  }

  return tokens;
}

function extractColorTokensFromTSX(filePath: string): Set<string> {
  const content = readFileSync(filePath, "utf-8");
  const tokens = new Set<string>();

  const classNameMatches = content.matchAll(/className=["']([^"']+)["']/g);
  for (const match of classNameMatches) {
    const classNames = match[1];
    for (const prefix of COLOR_PREFIXES) {
      const re = new RegExp(`${prefix}([a-zA-Z][a-zA-Z0-9-]*)`, "g");
      const tokenMatches = classNames.matchAll(re);
      for (const tokenMatch of tokenMatches) {
        const value = tokenMatch[1];
        if (value.includes("[") || value.includes("]")) continue;
        if (value.startsWith("linear-")) continue;
        if (value.startsWith("to-")) continue;
        if (value.includes("-to-")) continue;
        if (/^[tblrxy]-/.test(value)) continue;
        tokens.add(value);
      }
    }
  }

  return tokens;
}

describe("design tokens — structural", () => {
  test("Montserrat font is imported", () => {
    expect(css).toContain("Montserrat");
    expect(css).toContain("fonts.googleapis.com");
  });

  test("@utility typography scale is defined", () => {
    const textTokens = ["text-display", "text-h1", "text-h2", "text-h3", "text-body", "text-body-compact", "text-caption-bold", "text-caption", "text-small", "text-button"];
    for (const token of textTokens) {
      expect(css).toContain(`@utility ${token}`);
    }
  });

  test("@keyframes glimmer is defined", () => {
    expect(css).toContain("@keyframes glimmer");
    expect(css).toContain("0%, 100% { opacity: 0.25; }");
    expect(css).toContain("50%       { opacity: 1.0; }");
  });

  test(".glimmer-placeholder is defined", () => {
    expect(css).toContain(".glimmer-placeholder");
    expect(css).toContain("background-color: var(--color-disabled-text)");
    expect(css).toContain("border-radius: var(--radius-sm)");
  });

  test(".glass is defined with backdrop-filter", () => {
    expect(css).toContain(".glass");
    expect(css).toContain("backdrop-filter: blur(12px)");
  });

  test("spacing scale --space-1 through --space-14 defined", () => {
    for (let i = 1; i <= 14; i++) {
      expect(css).toContain(`--space-${i}:`);
    }
  });

  test("radius tokens defined", () => {
    expect(css).toContain("--radius-sm:");
    expect(css).toContain("--radius-card:");
    expect(css).toContain("--radius-feature:");
    expect(css).toContain("--radius-pill:");
  });

  test("shadow tokens defined", () => {
    expect(css).toContain("--shadow-level-1:");
    expect(css).toContain("--shadow-level-2:");
  });
});

describe("design tokens — cross-reference guard", () => {
  const definedTokens = extractTokensFromTheme(css);

  test("all custom color/border/bg/divide tokens used in TSX files are defined in @theme", () => {
    const srcDir = resolve(__dirname, "../components");
    const pagesDir = resolve(__dirname, "../pages");

    const undefinedTokensByFile: Record<string, string[]> = {};

    function checkDir(dir: string) {
      const files = readdirSync(dir, { withFileTypes: true });
      for (const file of files) {
        const fullPath = resolve(dir, file.name);
        if (file.isDirectory()) {
          checkDir(fullPath);
        } else if (file.name.endsWith(".tsx")) {
          const tsxTokens = extractColorTokensFromTSX(fullPath);
          const undefinedTokens: string[] = [];
          for (const token of tsxTokens) {
            const isMultiWord = token.includes("-");
            const isBuiltIn = TAILWIND_BUILT_IN_COLORS.has(token);
            const isDefined = definedTokens.has(token);
            if (isMultiWord && !isBuiltIn && !isDefined) {
              undefinedTokens.push(token);
            }
          }
          if (undefinedTokens.length > 0) {
            undefinedTokensByFile[fullPath] = undefinedTokens;
          }
        }
      }
    }

    checkDir(srcDir);
    checkDir(pagesDir);

    const entries = Object.entries(undefinedTokensByFile);
    if (entries.length > 0) {
      const msg = entries.map(([file, tokens]) => `${file}: ${tokens.join(", ")}`).join("\n");
      expect.fail(`Undefined tokens found:\n${msg}`);
    }
  });
});
