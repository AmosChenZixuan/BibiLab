import { afterEach, describe, expect, test, vi } from "vitest";

import { createApiClient } from "@/lib/api";
import { LANG_STORAGE_KEY, getUiLang, setUiLang } from "@/lib/utils";
import { mockFetch } from "@/test/utils";

afterEach(() => {
  localStorage.removeItem(LANG_STORAGE_KEY);
  vi.restoreAllMocks();
});

describe("getUiLang", () => {
  test("returns 'en' when localStorage has no entry", () => {
    localStorage.removeItem(LANG_STORAGE_KEY);
    expect(getUiLang()).toBe("en");
  });

  test("returns 'en' when localStorage has a non-zh value", () => {
    localStorage.setItem(LANG_STORAGE_KEY, "fr");
    expect(getUiLang()).toBe("en");
  });

  test("returns 'zh' when localStorage is 'zh'", () => {
    localStorage.setItem(LANG_STORAGE_KEY, "zh");
    expect(getUiLang()).toBe("zh");
  });

  test("reflects setUiLang writes", () => {
    setUiLang("zh");
    expect(getUiLang()).toBe("zh");
    setUiLang("en");
    expect(getUiLang()).toBe("en");
  });
});

describe("setUiLang", () => {
  test("writes the value to localStorage", () => {
    setUiLang("zh");
    expect(localStorage.getItem(LANG_STORAGE_KEY)).toBe("zh");
  });
});

describe("api request shape picks up setUiLang", () => {
  test("request built after setUiLang('zh') carries X-UI-Lang: zh", async () => {
    setUiLang("zh");
    const fetchMock = mockFetch(async () =>
      Response.json({ ok: true }, { status: 200 }),
    );

    const client = createApiClient("http://localhost:8765/api");
    await client.getHealth();

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ "X-UI-Lang": "zh" }),
      }),
    );

    setUiLang("en");
  });
});
