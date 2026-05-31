import { useCallback, useState } from "react";
import { ApiError } from "@/lib/api";

export function usePendingDeletions() {
  const [pending, setPending] = useState<Set<string>>(new Set());
  const isPending = useCallback((id: string) => pending.has(id), [pending]);

  const run = useCallback(async (id: string, mutate: () => Promise<void>): Promise<boolean> => {
    let started = false;
    setPending((prev) => {
      if (prev.has(id)) return prev;
      started = true;
      return new Set(prev).add(id);
    });
    if (!started) return false;

    try {
      await mutate();
      return true;
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return false;
      throw e;
    } finally {
      setPending((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }, []);

  return { isPending, run };
}
