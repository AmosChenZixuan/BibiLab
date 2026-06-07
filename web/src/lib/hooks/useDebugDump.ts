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

    // `run_chat_turn` writes the dump to disk before it emits the turn's
    // `done` SSE event, so by the time the user can click `</>` the file is
    // already present. A 404 therefore means the best-effort write failed,
    // not a race — surface it as notFound rather than retrying.
    api
      .getDebugDump(messageId)
      .then((d) => {
        if (!cancelled) setDump(d);
      })
      .catch((err) => {
        if (cancelled) return;
        if ((err as ApiError).status === 404) setNotFound(true);
        else console.error("debug dump fetch failed", err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

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
