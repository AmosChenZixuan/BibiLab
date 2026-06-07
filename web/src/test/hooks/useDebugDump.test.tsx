import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  const { createMockApi } = await import("@/test/utils");
  return {
    ...actual,
    api: createMockApi({
      getDebugDump: vi.fn(),
      getHealth: vi.fn(),
      getConfig: vi.fn(),
      putConfig: vi.fn(),
      listLists: vi.fn(),
      createList: vi.fn(),
      updateList: vi.fn(),
      deleteList: vi.fn(),
      createArtifact: vi.fn(),
      listSources: vi.fn(),
      getSource: vi.fn(),
      deleteSource: vi.fn(),
      rerunDigest: vi.fn(),
      updateSourceFacets: vi.fn(),
      previewPlaylist: vi.fn(),
      previewPlaylistMetadata: vi.fn(),
      ingestUrl: vi.fn(),
      listArtifacts: vi.fn(),
      getArtifactContent: vi.fn(),
      updateArtifact: vi.fn(),
      deleteArtifact: vi.fn(),
      listJobs: vi.fn(),
      deleteJob: vi.fn(),
      listModels: vi.fn(),
      downloadModel: vi.fn(),
      syncModels: vi.fn(),
      getConversation: vi.fn(),
      deleteConversation: vi.fn(),
    }),
  };
});

import { ApiError, api } from "@/lib/api";
import { useDebugDump } from "@/lib/hooks/useDebugDump";

describe("useDebugDump", () => {
  it("fetches dump on first call", async () => {
    (api.getDebugDump as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ messages: [] });
    const { result } = renderHook(() => useDebugDump("msg_1"));
    await waitFor(() => expect(result.current.dump).toEqual({ messages: [] }));
    expect(api.getDebugDump).toHaveBeenCalledWith("msg_1");
  });

  it("reports notFound on 404 (ApiError status 404)", async () => {
    // The dump is written before the turn's `done` event, so a 404 means the
    // best-effort write failed — surfaced directly as notFound, no retry.
    (api.getDebugDump as ReturnType<typeof vi.fn>).mockRejectedValue(
      new ApiError(404, "not found"),
    );
    const { result } = renderHook(() => useDebugDump("msg_missing"));
    await waitFor(() => expect(result.current.notFound).toBe(true));
    expect(result.current.dump).toBeNull();
  });
});
