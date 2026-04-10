import { afterEach, describe, expect, test, vi } from "vitest";

import { api } from "@/lib/api";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("api.getArtifactContent", () => {
  test("fetches artifact content from GET /artifacts/{id}/content", async () => {
    const fetchMock = vi.fn(async () =>
      new Response("# Artifact Content\n\nHere is the content.", {
        headers: { "Content-Type": "text/markdown" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.getArtifactContent("artifact-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:3000/api/artifacts/artifact-1/content",
      expect.objectContaining({ method: "GET" }),
    );
  });
});

describe("api.updateArtifact", () => {
  test("sends PATCH to /artifacts/{id} with name patch", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        id: "artifact-1",
        name: "New Name",
        type: "brief",
        prompt: "Generate a brief",
        source_ids: ["source-1"],
        status: "done",
        created_at: "2026-04-08T12:00:00Z",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.updateArtifact("artifact-1", { name: "New Name" });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/artifacts\/artifact-1/),
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ name: "New Name" }),
      }),
    );
  });
});

describe("api.deleteArtifact", () => {
  test("sends DELETE to /artifacts/{id}", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await api.deleteArtifact("artifact-1");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/artifacts\/artifact-1/),
      expect.objectContaining({ method: "DELETE" }),
    );
  });
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

    // api singleton uses window.location.origin as base URL (http://localhost in Vitest)
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/lists\/list-1$/),
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ thumbnail_source_id: "source-1" }),
      }),
    );
  });
});
