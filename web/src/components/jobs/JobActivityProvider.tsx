import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { api, toErrorMessage } from "../../lib/api";
import type { Job } from "../../lib/types";

export const TERMINAL_JOB_STATUSES = new Set(["done", "failed", "needs_auth"]);

type JobProducer = "ingest" | "whisper_download";

type JobRegistration = {
  id: string;
  producer: JobProducer;
  label: string;
  contextKey?: string;
};

export type JobActivityItem = {
  job: Job;
  producer: JobProducer;
  label: string;
  contextKey: string | null;
  isTerminal: boolean;
};

type TrackedJobMeta = {
  producer: JobProducer;
  label: string;
  contextKey: string | null;
};

type JobActivityContextValue = {
  activeJobs: JobActivityItem[];
  visibleJobs: JobActivityItem[];
  isPanelOpen: boolean;
  isPolling: boolean;
  cancellingJobId: string | null;
  errorMessage: string | null;
  clearTerminalJobs: () => void;
  dismissJob: (jobId: string) => void;
  cancelJob: (jobId: string) => Promise<void>;
  refreshNow: () => Promise<void>;
  setPanelOpen: (open: boolean) => void;
  trackJobs: (jobs: JobRegistration[]) => void;
  getJobs: (producer: JobProducer, contextKey?: string) => JobActivityItem[];
};

const JobActivityContext = createContext<JobActivityContextValue | null>(null);

function createPlaceholderJob(id: string, label: string): Job {
  return {
    id,
    type: "tracked",
    source_url: label,
    platform: "local",
    status: "queued",
    progress: 0,
    error: null,
    created_at: "",
    updated_at: "",
    meta: { title: label },
  };
}

function mergeJobs(
  previous: Record<string, Job>,
  trackedJobs: Record<string, TrackedJobMeta>,
  nextJobs: Job[],
) {
  const byId = new Map(nextJobs.map((job) => [job.id, job]));
  const merged: Record<string, Job> = {};

  for (const [jobId, meta] of Object.entries(trackedJobs)) {
    merged[jobId] = byId.get(jobId) ?? previous[jobId] ?? createPlaceholderJob(jobId, meta.label);
  }

  return merged;
}

function toActivityItem(job: Job, meta: TrackedJobMeta): JobActivityItem {
  return {
    job,
    producer: meta.producer,
    label: meta.label,
    contextKey: meta.contextKey,
    isTerminal: TERMINAL_JOB_STATUSES.has(job.status),
  };
}

export function getJobTitle(job: Job, fallbackLabel: string): string {
  const title = job.meta.title;
  if (typeof title === "string" && title.trim()) {
    return title;
  }

  const modelSize = job.meta.model_size;
  if (typeof modelSize === "string" && modelSize.trim()) {
    return modelSize;
  }

  if (job.source_url.trim()) {
    return job.source_url;
  }

  return fallbackLabel;
}

export function getJobTone(job: Job): "ok" | "error" | "unavailable" | "neutral" {
  if (job.status === "done") {
    return "ok";
  }
  if (job.status === "failed" || job.status === "needs_auth") {
    return "error";
  }
  if (job.status === "queued") {
    return "neutral";
  }
  return "unavailable";
}

export function JobActivityProvider({ children }: { children: React.ReactNode }) {
  const [jobsById, setJobsById] = useState<Record<string, Job>>({});
  const [trackedJobs, setTrackedJobs] = useState<Record<string, TrackedJobMeta>>({});
  const [dismissedJobIds, setDismissedJobIds] = useState<string[]>([]);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [cancellingJobId, setCancellingJobId] = useState<string | null>(null);
  const refreshInFlight = useRef<Promise<void> | null>(null);
  const trackedJobsRef = useRef<Record<string, TrackedJobMeta>>({});

  useEffect(() => {
    trackedJobsRef.current = trackedJobs;
  }, [trackedJobs]);

  const refreshNow = useCallback(async () => {
    if (refreshInFlight.current) {
      return refreshInFlight.current;
    }

    const task = (async () => {
      try {
        const nextJobs = await api.listJobs();
        setJobsById((current) => mergeJobs(current, trackedJobsRef.current, nextJobs));
        setErrorMessage(null);
      } catch (error) {
        setErrorMessage(toErrorMessage(error));
      } finally {
        refreshInFlight.current = null;
      }
    })();

    refreshInFlight.current = task;
    return task;
  }, []);

  const visibleJobs = useMemo(() => {
    return Object.entries(trackedJobs)
      .filter(([jobId]) => !dismissedJobIds.includes(jobId))
      .map(([jobId, meta]) => toActivityItem(jobsById[jobId] ?? createPlaceholderJob(jobId, meta.label), meta))
      .sort((left, right) => {
        if (left.isTerminal !== right.isTerminal) {
          return left.isTerminal ? 1 : -1;
        }
        return right.job.updated_at.localeCompare(left.job.updated_at);
      });
  }, [dismissedJobIds, jobsById, trackedJobs]);

  const activeJobs = useMemo(
    () => visibleJobs.filter((job) => !job.isTerminal),
    [visibleJobs],
  );

  useEffect(() => {
    if (activeJobs.length === 0) {
      return;
    }

    void refreshNow();
    const intervalId = window.setInterval(() => {
      void refreshNow();
    }, 5000);

    return () => window.clearInterval(intervalId);
  }, [activeJobs.length, refreshNow]);

  const trackJobs = useCallback((jobs: JobRegistration[]) => {
    if (jobs.length === 0) {
      return;
    }

    setTrackedJobs((current) => {
      const next = { ...current };
      for (const job of jobs) {
        next[job.id] = {
          producer: job.producer,
          label: job.label,
          contextKey: job.contextKey ?? null,
        };
      }
      trackedJobsRef.current = next;
      return next;
    });

    setDismissedJobIds((current) => current.filter((jobId) => !jobs.some((job) => job.id === jobId)));
    setJobsById((current) => {
      const next = { ...current };
      for (const job of jobs) {
        next[job.id] = current[job.id] ?? createPlaceholderJob(job.id, job.label);
      }
      return next;
    });

    void refreshNow();
  }, [refreshNow]);

  const dismissJob = useCallback((jobId: string) => {
    setDismissedJobIds((current) => (current.includes(jobId) ? current : [...current, jobId]));
    setTrackedJobs((current) => {
      const next = { ...current };
      delete next[jobId];
      trackedJobsRef.current = next;
      return next;
    });
    setJobsById((current) => {
      const next = { ...current };
      delete next[jobId];
      return next;
    });
  }, []);

  const clearTerminalJobs = useCallback(() => {
    const terminalIds = visibleJobs.filter((job) => job.isTerminal).map((job) => job.job.id);
    if (terminalIds.length === 0) {
      return;
    }

    setDismissedJobIds((current) => [...current, ...terminalIds.filter((jobId) => !current.includes(jobId))]);
    setTrackedJobs((current) => {
      const next = { ...current };
      for (const jobId of terminalIds) {
        delete next[jobId];
      }
      trackedJobsRef.current = next;
      return next;
    });
    setJobsById((current) => {
      const next = { ...current };
      for (const jobId of terminalIds) {
        delete next[jobId];
      }
      return next;
    });
  }, [visibleJobs]);

  const cancelJob = useCallback(async (jobId: string) => {
    setCancellingJobId(jobId);
    try {
      await api.deleteJob(jobId);
      dismissJob(jobId);
    } catch (error) {
      setErrorMessage(toErrorMessage(error));
    } finally {
      setCancellingJobId(null);
    }
  }, [dismissJob]);

  const getJobs = useCallback(
    (producer: JobProducer, contextKey?: string) => {
      return visibleJobs.filter(
        (item) =>
          item.producer === producer &&
          (contextKey === undefined ? true : item.contextKey === contextKey),
      );
    },
    [visibleJobs],
  );

  const value = useMemo<JobActivityContextValue>(() => ({
    activeJobs,
    visibleJobs,
    isPanelOpen,
    isPolling: activeJobs.length > 0,
    cancellingJobId,
    errorMessage,
    clearTerminalJobs,
    dismissJob,
    cancelJob,
    refreshNow,
    setPanelOpen: setIsPanelOpen,
    trackJobs,
    getJobs,
  }), [
    activeJobs,
    cancellingJobId,
    cancelJob,
    clearTerminalJobs,
    dismissJob,
    errorMessage,
    getJobs,
    isPanelOpen,
    refreshNow,
    trackJobs,
    visibleJobs,
  ]);

  return <JobActivityContext.Provider value={value}>{children}</JobActivityContext.Provider>;
}

export function useJobActivity() {
  const value = useContext(JobActivityContext);
  if (!value) {
    throw new Error("useJobActivity must be used inside JobActivityProvider");
  }
  return value;
}
