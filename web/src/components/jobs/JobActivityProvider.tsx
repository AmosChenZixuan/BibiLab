import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { api, toErrorMessageWithT } from "@/lib/api";
import type { ArtifactJob, IngestJob, Job, ModelDownloadJob } from "@/lib/types";
import { useLanguage } from "@/app/LanguageContext";

export const TERMINAL_JOB_STATUSES = new Set(["done", "failed", "needs_auth"]);

const POLL_INTERVAL_MS = 5_000;

type JobProducer = "ingest" | "whisper_download" | "artifact";

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
  dismissJob: (jobId: string) => Promise<void>;
  cancelJob: (jobId: string) => Promise<void>;
  refreshNow: () => Promise<void>;
  setPanelOpen: (open: boolean) => void;
  trackJobs: (jobs: JobRegistration[]) => void;
  getJobs: (producer: JobProducer, contextKey?: string) => JobActivityItem[];
};

const JobActivityContext = createContext<JobActivityContextValue | null>(null);

function createPlaceholderJob(id: string, meta: TrackedJobMeta): Job {
  if (meta.producer === "whisper_download") {
    return {
      id,
      type: "model_download",
      status: "queued",
      progress: 0,
      error: null,
      created_at: "",
      updated_at: "",
      meta: { model_family: "whisper", model_size: meta.label },
    };
  }

  if (meta.producer === "artifact") {
    return {
      id,
      type: "artifact",
      status: "generating",
      progress: 0,
      error: null,
      created_at: "",
      updated_at: "",
      meta: {
        list_id: meta.contextKey ?? undefined,
        artifact_id: id,
        artifact_type: meta.label,
      },
    };
  }

  return {
    id,
    type: "ingest",
    status: "queued",
    progress: 0,
    error: null,
    created_at: "",
    updated_at: "",
    meta: {
      title: meta.label,
      list_id: meta.contextKey ?? undefined,
      platform: "local",
      source_url: meta.label,
    },
  };
}

function isIngestJob(job: Job): job is IngestJob {
  return job.type === "ingest";
}

function isModelDownloadJob(job: Job): job is ModelDownloadJob {
  return job.type === "model_download";
}

function isArtifactJob(job: Job): job is ArtifactJob {
  return job.type === "artifact";
}

function extractIngestTitle(meta: Job["meta"]): string {
  if ("title" in meta && typeof meta.title === "string" && meta.title.trim()) {
    return meta.title;
  }
  if ("source_url" in meta && typeof meta.source_url === "string" && meta.source_url.trim()) {
    return meta.source_url;
  }
  return "Queued source";
}

function inferTrackedMeta(job: Job): TrackedJobMeta {
  if (isModelDownloadJob(job)) {
    const modelSize = typeof job.meta.model_size === "string" ? job.meta.model_size : "model";
    return {
      producer: "whisper_download",
      label: modelSize,
      contextKey: modelSize,
    };
  }

  if (isArtifactJob(job)) {
    const artifactType = typeof job.meta.artifact_type === "string" ? job.meta.artifact_type : "artifact";
    return {
      producer: "artifact",
      label: artifactType,
      contextKey: typeof job.meta.list_id === "string" ? job.meta.list_id : null,
    };
  }

  return {
    producer: "ingest",
    label: isIngestJob(job) ? extractIngestTitle(job.meta) : "Queued source",
    contextKey: isIngestJob(job) && typeof job.meta.list_id === "string" ? job.meta.list_id : null,
  };
}

function mergeJobs(
  previous: Record<string, Job>,
  trackedJobs: Record<string, TrackedJobMeta>,
  nextJobs: Job[],
) {
  const merged: Record<string, Job> = Object.fromEntries(
    nextJobs.map((job) => [job.id, job]),
  );

  for (const [jobId, meta] of Object.entries(trackedJobs)) {
    merged[jobId] = merged[jobId] ?? previous[jobId] ?? createPlaceholderJob(jobId, meta);
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
  if (isIngestJob(job)) {
    const title = extractIngestTitle(job.meta);
    if (title !== "Queued source") {
      return title;
    }
  }

  if (isModelDownloadJob(job) && typeof job.meta.model_size === "string" && job.meta.model_size.trim()) {
    return job.meta.model_size;
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
  const { t } = useLanguage();
  const [jobsById, setJobsById] = useState<Record<string, Job>>({});
  const [trackedJobs, setTrackedJobs] = useState<Record<string, TrackedJobMeta>>({});
  const [dismissedJobIds, setDismissedJobIds] = useState<string[]>([]);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [cancellingJobId, setCancellingJobId] = useState<string | null>(null);
  const trackedJobsRef = useRef<Record<string, TrackedJobMeta>>({});

  useEffect(() => {
    trackedJobsRef.current = trackedJobs;
  }, [trackedJobs]);

  const refreshNow = useCallback(async (signal?: AbortSignal) => {
    try {
      const nextJobs = await api.listJobs({ signal });
      setJobsById((current) => mergeJobs(current, trackedJobsRef.current, nextJobs ?? []));
      setErrorMessage(null);
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") return;
      setErrorMessage(toErrorMessageWithT(error, t));
    }
  }, [t]);

  useEffect(() => {
    const controller = new AbortController();
    void refreshNow(controller.signal);
    return () => controller.abort();
  }, [refreshNow]);

  const visibleJobs = useMemo(() => {
    return Object.values(jobsById)
      .filter((job) => !dismissedJobIds.includes(job.id))
      .map((job) => toActivityItem(job, trackedJobs[job.id] ?? inferTrackedMeta(job)))
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

    const controller = new AbortController();

    void refreshNow(controller.signal);
    const intervalId = window.setInterval(() => {
      void refreshNow(controller.signal);
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(intervalId);
      controller.abort();
    };
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

    const newJobIds = new Set(jobs.map((job) => job.id));
    setDismissedJobIds((current) => current.filter((jobId) => !newJobIds.has(jobId)));
    setJobsById((current) => {
      const next = { ...current };
      for (const job of jobs) {
        next[job.id] = current[job.id] ?? createPlaceholderJob(job.id, {
          producer: job.producer,
          label: job.label,
          contextKey: job.contextKey ?? null,
        });
      }
      return next;
    });

    void refreshNow();
  }, [refreshNow]);

  const removeJobLocally = useCallback((jobId: string) => {
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

  const dismissJob = useCallback(async (jobId: string) => {
    try {
      await api.deleteJob(jobId);
      removeJobLocally(jobId);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(toErrorMessageWithT(error, t));
    }
  }, [removeJobLocally, t]);

  const clearTerminalJobs = useCallback(() => {
    const terminalIds = visibleJobs.filter((job) => job.isTerminal).map((job) => job.job.id);
    if (terminalIds.length === 0) {
      return;
    }

    void Promise.all(terminalIds.map((jobId) => dismissJob(jobId)));
  }, [dismissJob, visibleJobs]);

  const cancelJob = useCallback(async (jobId: string) => {
    setCancellingJobId(jobId);
    try {
      await api.deleteJob(jobId);
      removeJobLocally(jobId);
    } catch (error) {
      setErrorMessage(toErrorMessageWithT(error, t));
    } finally {
      setCancellingJobId(null);
    }
  }, [removeJobLocally, t]);

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
