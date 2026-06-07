import { useEffect, useState } from "react";

import { api, ApiError } from "@/lib/api";

export function useDebugDump(messageId: string | null) {
  const [dump, setDump] = useState<unknown | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!messageId) return;
    let cancelled = false;
    setDump(null);
    setLoading(true);
    setNotFound(false);

    // Backend writes the dump at end of `run_chat_turn` which runs AFTER
    // the `done` SSE event is yielded. If the user clicks `</>` immediately
    // after stream end, the file may not be on disk yet — race a 404.
    // Retry once after a short delay to absorb the race.
    const fetchWithRetry = async (attempt: number): Promise<void> => {
      try {
        const d = await api.getDebugDump(messageId);
        if (cancelled) return;
        setDump(d);
      } catch (err) {
        if (cancelled) return;
        const status = (err as ApiError).status;
        if (status === 404 && attempt < 3) {
          // Backoff: 200ms, 600ms
          await new Promise((r) => setTimeout(r, 200 * attempt ** 2));
          if (cancelled) return;
          return fetchWithRetry(attempt + 1);
        }
        if (status === 404) setNotFound(true);
        else console.error("debug dump fetch failed", err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void fetchWithRetry(1);

    return () => {
      cancelled = true;
    };
  }, [messageId]);

  // Synchronous reset — call from the open handler (e.g. on `</>` click)
  // to clear stale state before the next render. Without this, the drawer
  // briefly renders the previous turn's dump because React batches the
  // setMsgId call but the effect's setDump(null) only runs after commit.
  const reset = () => {
    setDump(null);
    setNotFound(false);
    setLoading(false);
  };

  return { dump, notFound, loading, reset };
}
