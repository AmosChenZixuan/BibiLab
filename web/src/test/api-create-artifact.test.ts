import { afterEach, describe, expect, test, vi } from "vitest";
import { createApiClient } from "@/lib/api";
import { setUiLang } from "@/lib/utils";
import { mockFetch } from "@/test/utils";

describe("ListsClient.createArtifact", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("createArtifact is a function on ListsClient", () => {
    const client = createApiClient("http://localhost:8765/api");
    expect(typeof client.createArtifact).toBe("function");
  });

  test("createArtifact calls POST /api/lists/{listId}/artifacts with correct body", async () => {
    const fetchMock = mockFetch(async () =>
      Response.json(
        {
          id: "job-123",
          type: "ingest",
          status: "queued",
          progress: 0,
          error: null,
          created_at: "2026-04-08T00:00:00Z",
          updated_at: "2026-04-08T00:00:00Z",
          meta: { list_id: "list-1" },
        },
        { status: 201 },
      ),
    );

    const client = createApiClient("http://localhost:8765/api");
    const result = await client.createArtifact("list-1", {
      type: "brief",
      prompt: "Give me a brief",
      source_ids: ["src-1", "src-2"],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8765/api/lists/list-1/artifacts",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Content-Type": "application/json", "X-UI-Lang": "en" }),
        body: JSON.stringify({
          type: "brief",
          prompt: "Give me a brief",
          source_ids: ["src-1", "src-2"],
        }),
      }),
    );
    expect(result).toMatchObject({ id: "job-123" });
  });

  test("createArtifact returns Promise<Job>", async () => {
    const fetchMock = mockFetch(async () =>
      Response.json(
        {
          id: "job-456",
          type: "ingest",
          status: "queued",
          progress: 0,
          error: null,
          created_at: "2026-04-08T00:00:00Z",
          updated_at: "2026-04-08T00:00:00Z",
          meta: { list_id: "list-1" },
        },
        { status: 201 },
      ),
    );

    const client = createApiClient("http://localhost:8765/api");
    const result = await client.createArtifact("list-1", {
      type: "custom_report",
      prompt: "Custom analysis",
      source_ids: ["src-1"],
    });

    expect(result.id).toBe("job-456");
    expect(result.status).toBe("queued");
  });

  test("setUiLang('zh') updates X-UI-Lang header to zh", async () => {
    const fetchMock = mockFetch(async () =>
      Response.json(
        {
          id: "job-789",
          type: "artifact",
          status: "queued",
          progress: 0,
          error: null,
          created_at: "2026-04-08T00:00:00Z",
          updated_at: "2026-04-08T00:00:00Z",
          meta: {},
        },
        { status: 201 },
      ),
    );

    setUiLang("zh");
    const client = createApiClient("http://localhost:8765/api");
    await client.createArtifact("list-1", {
      type: "brief",
      prompt: "Brief",
      source_ids: ["src-1"],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8765/api/lists/list-1/artifacts",
      expect.objectContaining({
        headers: expect.objectContaining({ "X-UI-Lang": "zh" }),
      }),
    );

    setUiLang("en");
  });
});
