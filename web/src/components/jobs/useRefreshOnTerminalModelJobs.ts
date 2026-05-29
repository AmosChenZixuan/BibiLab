import { useEffect } from "react";

import { useJobActivity } from "./JobActivityProvider";

const MODEL_DOWNLOAD_PRODUCER = "model_download";

/**
 * Refresh + dismiss terminal model-download jobs. Depends on
 * terminalCount (primitive) instead of modelJobs array identity to avoid
 * effect-storm when JobActivityProvider returns a fresh array each render.
 */
export function useRefreshOnTerminalModelJobs(refresh: () => Promise<void>): void {
  const { getJobs, dismissJob } = useJobActivity();
  const modelJobs = getJobs(MODEL_DOWNLOAD_PRODUCER);
  const terminalCount = modelJobs.filter((j) => j.isTerminal).length;

  useEffect(() => {
    if (terminalCount === 0) return;
    const terminalIds = modelJobs.filter((j) => j.isTerminal).map((j) => j.job.id);
    async function refreshAndDismiss() {
      await refresh();
      for (const id of terminalIds) dismissJob(id);
    }
    void refreshAndDismiss();
  }, [modelJobs, terminalCount, dismissJob, refresh]);
}
