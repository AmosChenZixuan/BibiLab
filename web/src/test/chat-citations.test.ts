import { describe, expect, test } from "vitest";
import { stripLegacyTokens, type ContentBlock } from "@/lib/chat-utils";

describe("stripLegacyTokens", () => {
  test("removes [Title @ Ns-Ns] patterns", () => {
    expect(stripLegacyTokens("See [My Video @ 120s-145s].")).toBe("See .");
  });

  test("removes multiple", () => {
    expect(stripLegacyTokens("[A @ 10s-20s] and [B @ 30s-40s]")).toBe(" and ");
  });

  test("leaves non-citation brackets intact", () => {
    expect(stripLegacyTokens("array[0] and [note]")).toBe("array[0] and [note]");
  });

  test("handles empty string", () => {
    expect(stripLegacyTokens("")).toBe("");
  });
});

describe("ContentBlock types", () => {
  test("text block shape", () => {
    const b: ContentBlock = { type: "text", text: "hello" };
    if (b.type === "text") {
      expect(typeof b.text).toBe("string");
    }
  });

  test("citation block shape", () => {
    const b: ContentBlock = { type: "citation", index: 1, source_id: "s1", chunk_ids: ["c1"] };
    if (b.type === "citation") {
      expect(b.index).toBe(1);
      expect(b.source_id).toBe("s1");
      expect(b.chunk_ids).toEqual(["c1"]);
    }
  });
});
