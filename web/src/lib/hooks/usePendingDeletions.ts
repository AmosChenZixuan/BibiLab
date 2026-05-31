import { useCallback, useRef, useState } from "react";
import { ApiError } from "@/lib/api";

export function usePendingDeletions() {
  // Ref is the authoritative in-flight set (synchronous read for dedupe);
  // state mirrors it only to drive isPending re-renders.
  const pendingRef = useRef<Set<string>>(new Set());
  const [pending, setPending] = useState<Set<string>>(new Set());

  const isPending = useCallback((id: string) => pending.has(id), [pending]);

  const run = useCallback(async (id: string, mutate: () => Promise<void>): Promise<void> => {
    if (pendingRef.current.has(id)) return; // already in flight -- dedupe
    pendingRef.current.add(id);
    setPending(new Set(pendingRef.current));

    try {
      await mutate();
    } catch (e) {
      if (!(e instanceof ApiError && e.status === 404)) throw e; // 404 = already gone, ok
    } finally {
      pendingRef.current.delete(id);
      setPending(new Set(pendingRef.current));
    }
  }, []);

  return { isPending, run };
}
