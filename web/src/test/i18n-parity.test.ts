import { describe, expect, test } from "vitest";

import en from "@/lib/i18n/en.json";
import zh from "@/lib/i18n/zh.json";

function flattenKeys(obj: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(obj).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return value && typeof value === "object" && !Array.isArray(value)
      ? flattenKeys(value as Record<string, unknown>, path)
      : [path];
  });
}

describe("i18n string tables", () => {
  test("en and zh key sets are identical", () => {
    expect(flattenKeys(zh).sort()).toEqual(flattenKeys(en).sort());
  });
});
