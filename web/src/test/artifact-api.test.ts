import { afterEach, describe, expect, test, vi } from "vitest";

import { api } from "@/lib/api";
import { mockFetch } from "@/test/utils";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("api.listArtifacts", () => {
  test("calls GET /api/lists/{listId}/artifacts", async () => {
    const fetchMock = mockFetch(async () =>
      Response.json([
        {
          id: "artifact-1",
          name: "My Artifact",
          type: "brief",
          prompt: "Generate a brief",
          source_ids: ["source-1"],
          status: "done",
          created_at: "2026-04-08T12:00:00Z",
        },
      ]),
    );

    await api.listArtifacts("list-1");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/lists\/list-1\/artifacts$/),
      expect.objectContaining({ headers: expect.objectContaining({ "Content-Type": "application/json" }) }),
    );
  });

  test("returns array of artifacts", async () => {
    const fetchMock = mockFetch(async () =>
      Response.json([
        {
          id: "artifact-1",
          name: "Brief",
          type: "brief",
          prompt: "prompt",
          source_ids: [],
          status: "done",
          created_at: "2026-04-08T12:00:00Z",
        },
        {
          id: "artifact-2",
          name: "Study Guide",
          type: "study_guide",
          prompt: "prompt",
          source_ids: [],
          status: "generating",
          created_at: "2026-04-08T12:01:00Z",
        },
      ]),
    );

    const result = await api.listArtifacts("list-1");

    expect(result).toHaveLength(2);
    expect(result?.[0].id).toBe("artifact-1");
    expect(result?.[1].status).toBe("generating");
  });
});
