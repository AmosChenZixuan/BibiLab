import { afterEach, describe, expect, test, vi } from "vitest";

import { api } from "@/lib/api";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("api.updateList", () => {
  test("sends partial list updates including thumbnail_source_id", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        id: "list-1",
        name: "Systems",
        created_at: "2026-03-31T19:00:00Z",
        thumbnail_source_id: "source-1",
        thumbnail_url: "http://testserver/covers/source-1",
        source_count: 3,
        updated_at: "2026-03-31T20:00:00Z",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.updateList("list-1", { thumbnail_source_id: "source-1" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/lists/list-1",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ thumbnail_source_id: "source-1" }),
      }),
    );
  });
});
