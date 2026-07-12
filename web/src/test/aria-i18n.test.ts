/// <reference types="vite/client" />
import { describe, expect, test } from "vitest";

// Raw-source scan: every aria-label must go through t() like visible text.
// Screen-reader users in zh otherwise get English labels.
const sources = import.meta.glob("../**/*.tsx", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const HARDCODED_ARIA = [
  /aria-label="[A-Za-z]/, // aria-label="Close"
  /aria-label=\{`/, // aria-label={`Open ${title}`}
  /aria-label=\{[^}]*"(?:[A-Z][^"]*|[^"]* [^"]*)"/, // English string literal inside the expression
];

describe("aria-label i18n", () => {
  test("no hardcoded English aria-labels outside tests", () => {
    const offenders: string[] = [];
    for (const [path, content] of Object.entries(sources)) {
      if (path.includes(".test.")) continue;
      content.split("\n").forEach((line, i) => {
        // Blank out t("key") calls so their key strings can't false-positive.
        const cleaned = line.replace(/t\("[^"]+"(?:,\s*\{[^}]*\})?\)/g, "t()");
        if (HARDCODED_ARIA.some((re) => re.test(cleaned))) {
          offenders.push(`${path}:${i + 1} ${line.trim()}`);
        }
      });
    }
    expect(offenders).toEqual([]);
  });
});
