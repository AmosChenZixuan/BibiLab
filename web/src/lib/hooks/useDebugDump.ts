import { useEffect, useState } from "react";

import { api, ApiError } from "@/lib/api";

export function useDebugDump(messageId: string | null) {
  const [dump, setDump] = useState<unknown | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!messageId) return;
    let cancelled = false;
    setLoading(true);
    setNotFound(false);
    api
      .getDebugDump(messageId)
      .then((d) => {
        if (!cancelled) setDump(d);
      })
      .catch((err: ApiError) => {
        if (cancelled) return;
        if (err.status === 404) setNotFound(true);
        else console.error("debug dump fetch failed", err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [messageId]);

  return { dump, notFound, loading };
}
