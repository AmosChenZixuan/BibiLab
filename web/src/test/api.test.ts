import { afterEach, describe, expect, test, vi } from "vitest";

import { api, ApiError, toErrorMessageWithT } from "@/lib/api";

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

describe("toErrorMessageWithT", () => {
  const identity = (k: string) => k;

  test("returns string detail when ApiError has a non-empty string detail", () => {
    const error = new ApiError(400, "Unsupported URL: https://example.com/video");
    expect(toErrorMessageWithT(error, identity)).toBe("Unsupported URL: https://example.com/video");
  });

  test("returns detail.message when ApiError has object detail with message", () => {
    const error = new ApiError(400, { message: "'12' is not a valid URL" });
    expect(toErrorMessageWithT(error, identity)).toBe("'12' is not a valid URL");
  });

  test("returns errors.apiError when ApiError detail is empty object", () => {
    const error = new ApiError(500, {});
    expect(toErrorMessageWithT(error, identity)).toBe("errors.apiError");
  });

  test("returns errors.apiError when ApiError detail is empty string", () => {
    const error = new ApiError(400, "");
    expect(toErrorMessageWithT(error, identity)).toBe("errors.apiError");
  });

  test("returns errors.401 when ApiError is 401 with object detail", () => {
    const error = new ApiError(401, { message: "Unauthorized" });
    expect(toErrorMessageWithT(error, identity)).toBe("errors.401");
  });

  test("returns string detail when ApiError is 401 with string detail", () => {
    const error = new ApiError(401, "Token expired");
    expect(toErrorMessageWithT(error, identity)).toBe("Token expired");
  });
});

describe("api.getConfig", () => {
  test("getConfig returns username and avatar_url for bilibili account", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        accounts: {
          bilibili: {
            cookie: "***",
            last_verified: "2025-01-01T00:00:00Z",
            username: "test_user",
            avatar_url: "https://i0.hdslb.com/bfs/face/abc.jpg",
          },
        },
        ai: { protocol: "openai", model: "gpt-4o", api_key: "***", base_url: "", output_language: "ui" },
        transcription: { engine: "whisper", model_size: "large-v3", device: "cpu", language: "auto" },
        vision: { enabled: false, frame_sample_rate: 30, model: null },
        backend: { port: 8765, worker_concurrency: 1 },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const config = await api.getConfig();
    expect(config?.accounts.bilibili.username).toBe("test_user");
    expect(config?.accounts.bilibili.avatar_url).toBe("https://i0.hdslb.com/bfs/face/abc.jpg");
  });
});
