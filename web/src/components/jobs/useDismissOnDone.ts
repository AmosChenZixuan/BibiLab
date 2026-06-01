import { useEffect, useRef } from "react";

import {
  type JobActivityItem,
  useJobActivity,
} from "@/components/jobs/JobActivityProvider";
import type { Job } from "@/lib/types";

/**
 * Auto-dismiss jobs that have reached terminal `done` status.
 *
 * On every render where new `done` jobs appear, this runs `onDone(job)` for
 * each (sequentially), then calls `dismissJob` to remove it from the spirit.
 *
 * Dedup: jobs are added to a ref Set synchronously *before* any await, so a
 * subsequent effect run (e.g., from polling) sees them as already-handled and
 * skips — no race with the in-flight `dismissJob`. `dismissJob` is also
 * idempotent via `usePendingDeletions`, so letting an in-flight call finish
 * is safe even after the component unmounts.
 *
 * `onDone` may be a fresh function each render; the latest is always used.
 */
export function useDismissOnDone({
  jobs,
  onDone,
}: {
  jobs: JobActivityItem[];
  onDone: (job: Job) => Promise<void> | void;
}) {
  const { dismissJob } = useJobActivity();
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;
  const dismissedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const completed = jobs.filter(
      (item) => item.isTerminal && item.job.status === "done" && !dismissedRef.current.has(item.job.id),
    );
    if (completed.length === 0) return;

    // Sync dedup: a concurrent effect run (poll re-render) will see these as
    // already-handled and skip — avoids duplicate onDone / dismissJob calls.
    for (const item of completed) {
      dismissedRef.current.add(item.job.id);
    }

    void (async () => {
      for (const item of completed) {
        try {
          await onDoneRef.current(item.job);
        } catch (err) {
          console.error("useDismissOnDone: onDone callback failed", err);
        }
        await dismissJob(item.job.id);
      }
    })();
  }, [jobs, dismissJob]);
}
