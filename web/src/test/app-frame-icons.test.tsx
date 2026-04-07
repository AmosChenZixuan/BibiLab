import { beforeAll, describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe("AppFrame icon migration", () => {
  const appFramePath = join(__dirname, "../components/layout/AppFrame.tsx");
  let source: string;

  beforeAll(() => {
    source = readFileSync(appFramePath, "utf-8");
  });

  it("should NOT contain FiSettings import from react-icons/fi", () => {
    const fiSettingsPattern = /FiSettings/;
    expect(source).not.toMatch(fiSettingsPattern);
  });

  it("should NOT contain FiUser import from react-icons/fi", () => {
    const fiUserPattern = /FiUser/;
    expect(source).not.toMatch(fiUserPattern);
  });

  it("should import Settings from lucide-react", () => {
    const settingsPattern = /Settings.*from\s+["']lucide-react["']/;
    expect(source).toMatch(settingsPattern);
  });

  it("should import User from lucide-react", () => {
    const userPattern = /User.*from\s+["']lucide-react["']/;
    expect(source).toMatch(userPattern);
  });

  it("should use hover:bg-sky/10 on all three icon buttons", () => {
    const hoverSkyPattern = /hover:bg-sky\//g;
    const matches = source.match(hoverSkyPattern);
    expect(matches).toHaveLength(3);
  });
});
