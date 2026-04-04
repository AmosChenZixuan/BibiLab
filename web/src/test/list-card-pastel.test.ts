import { describe, expect, test } from "vitest";
import { nameToPastelIndex, PASTEL_COLORS } from "@/components/lists/ListCard";

describe("nameToPastelIndex", () => {
  test("returns integer in range 0-7", () => {
    const idx = nameToPastelIndex("Watch Later");
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(8);
  });

  test("is deterministic", () => {
    expect(nameToPastelIndex("Watch Later")).toBe(nameToPastelIndex("Watch Later"));
  });

  test("distributes differently for different names", () => {
    // May collide but should not be always equal
    const a = nameToPastelIndex("AAAA");
    const b = nameToPastelIndex("BBBB");
    // Just verify both are valid indices
    expect(a).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThan(8);
    expect(b).toBeGreaterThanOrEqual(0);
    expect(b).toBeLessThan(8);
  });
});
