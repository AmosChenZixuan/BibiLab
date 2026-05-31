import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import { ApiError } from "@/lib/api";
import { usePendingDeletions } from "@/lib/hooks/usePendingDeletions";

describe("usePendingDeletions", () => {
  test("dedupes a second call while the first is in flight", async () => {
    const { result } = renderHook(() => usePendingDeletions());

    let resolveFirst!: () => void;
    const mutate = vi.fn(() => new Promise<void>((resolve) => { resolveFirst = resolve; }));

    await act(async () => {
      void result.current.run("a", mutate);
    });
    expect(result.current.isPending("a")).toBe(true);
    expect(mutate).toHaveBeenCalledTimes(1);

    // Second call for the same id is a no-op while in flight.
    await act(async () => {
      void result.current.run("a", mutate);
    });
    expect(mutate).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveFirst();
    });
    await waitFor(() => expect(result.current.isPending("a")).toBe(false));
  });

  test("calls mutate and clears pending on success", async () => {
    const { result } = renderHook(() => usePendingDeletions());
    const mutate = vi.fn(() => Promise.resolve());

    await act(async () => {
      await result.current.run("a", mutate);
    });

    expect(mutate).toHaveBeenCalledTimes(1);
    expect(result.current.isPending("a")).toBe(false);
  });

  test("swallows a 404 (already gone) and resolves", async () => {
    const { result } = renderHook(() => usePendingDeletions());
    const mutate = () => Promise.reject(new ApiError(404, "gone"));

    await act(async () => {
      await expect(result.current.run("a", mutate)).resolves.toBeUndefined();
    });
    expect(result.current.isPending("a")).toBe(false);
  });

  test("rethrows a non-404 ApiError and clears pending", async () => {
    const { result } = renderHook(() => usePendingDeletions());
    const mutate = () => Promise.reject(new ApiError(500, "boom"));

    await act(async () => {
      await expect(result.current.run("a", mutate)).rejects.toBeInstanceOf(ApiError);
    });
    expect(result.current.isPending("a")).toBe(false);
  });

  test("rethrows a plain error and clears pending", async () => {
    const { result } = renderHook(() => usePendingDeletions());
    const mutate = () => Promise.reject(new Error("network"));

    await act(async () => {
      await expect(result.current.run("a", mutate)).rejects.toThrow("network");
    });
    expect(result.current.isPending("a")).toBe(false);
  });
});
