import { readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it } from "vitest";

// Guards the inlined favicon (index.html) against drift from its source
// (public/favicon.svg). The icon is inlined as a data: URI so the browser
// never requests /favicon.svg — which the single-port backend can't serve.
describe("inlined favicon", () => {
  it("matches public/favicon.svg byte-for-byte", () => {
    const html = readFileSync(resolve(__dirname, "../../index.html"), "utf8");
    const match = html.match(
      /<link[^>]+rel="icon"[^>]+href="data:image\/svg\+xml;base64,([^"]+)"/,
    );
    expect(match, "index.html must link the favicon as a base64 data URI").not.toBeNull();
    const inlined = Buffer.from(match![1], "base64");
    const source = readFileSync(resolve(__dirname, "../../public/favicon.svg"));
    expect(inlined.equals(source)).toBe(true);
  });
});
