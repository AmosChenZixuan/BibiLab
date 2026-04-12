import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowRight, X, Trash2, AlertCircle, MoreVertical } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { ContextMenu } from "@/components/ui/ContextMenu";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api, toErrorMessageWithT } from "@/lib/api";
import type { Source } from "@/lib/types";

export const PIPELINE_STAGES = [
  "queued",
  "downloading",
  "transcribing",
  "processing",
  "done",
] as const;
export type PipelineStage = (typeof PIPELINE_STAGES)[number];

function SourceRow({
  source,
  onOpen,
  onDelete,
  t,
}: {
  source: Source;
  onOpen: () => void;
  onDelete: () => Promise<void>;
  t: (key: string, params?: Record<string, string | number>) => string;
}) {
  return (
    <div className="group flex items-center gap-2 rounded-2xl border border-border bg-white/64 px-4 py-3 transition hover:bg-white hover:shadow-sm">
      <button
        type="button"
        aria-label={`Open ${source.title}`}
        className="min-w-0 flex-1 border-0 bg-transparent text-left"
        onClick={onOpen}
      >
        <p className="m-0 truncate text-sm font-medium text-ink">{source.title}</p>
        <p className="m-0 mt-0.5 text-xs text-muted">{source.platform}</p>
      </button>
      <ContextMenu
        items={[
          { label: t("lists.delete"), icon: <Trash2 />, onClick: onDelete, variant: "danger" },
        ]}
        trigger={({ toggle, triggerRef }) => (
          <button
            ref={triggerRef}
            type="button"
            aria-label="Source options"
            onClick={toggle}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted opacity-0 transition group-hover:opacity-100 hover:bg-border hover:text-ink"
          >
            <MoreVertical size={16} />
          </button>
        )}
      />
    </div>
  );
}

function IngestingSourceRow({
  stage,
  title,
  onDismiss,
  t,
}: {
  stage: string;
  title: string;
  onDismiss: () => void;
  t: (key: string, params?: Record<string, string | number>) => string;
}) {
  const isFailed = stage === "failed" || stage === "needs_auth";
  const stageIdx = isFailed ? -1 : PIPELINE_STAGES.indexOf(stage as PipelineStage);
  const displayStage = isFailed ? "failed" : (PIPELINE_STAGES[Math.max(0, stageIdx)] ?? "downloading");

  const getStageLabel = (s: string) =>
    s === "failed" ? t("pipeline.failed") : t("pipeline." + s);

  if (isFailed) {
    return (
      <div className="flex items-start gap-3 rounded-2xl border border-pink/30 bg-pink/6 px-4 py-3">
        <AlertCircle size={16} className="mt-0.5 shrink-0 text-pink" />
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-sm font-medium text-ink">{title}</p>
          <p className="m-0 mt-0.5 text-xs text-pink">{t("lists.failedDuring", { stage: getStageLabel(displayStage) })}</p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
        >
          <X size={14} />
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5 rounded-2xl border border-blue/20 bg-sky/6 px-4 py-3">
      <div className="flex items-center gap-2">
        <span className="relative flex h-2 w-2 shrink-0">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue/40" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-blue/70" />
        </span>
        <p className="m-0 min-w-0 flex-1 truncate text-sm font-medium text-ink">{title}</p>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Cancel ingestion"
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
        >
          <X size={14} />
        </button>
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs text-blue/80">{t("pipeline." + displayStage)}…</span>
          <span className="text-xs tabular-nums text-muted/60">
            {Math.max(1, stageIdx + 1)} / {PIPELINE_STAGES.length}
          </span>
        </div>
        <div className="flex gap-0.5">
          {PIPELINE_STAGES.map((s, i) => (
            <div
              key={s}
              title={t("pipeline." + s)}
              className={`h-1 flex-1 rounded-full transition-colors duration-500 ${
                i < stageIdx
                  ? "bg-blue/60"
                  : i === stageIdx
                    ? "animate-pulse bg-blue/90"
                    : "bg-border"
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export function SourcesListMode({
  listId,
  sources,
  onOpenSource,
}: {
  listId: string;
  sources: Source[];
  onOpenSource: (source: Source) => void;
}) {
  const { t } = useLanguage();
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { dismissJob, getJobs, trackJobs } = useJobActivity();
  const ingestJobs = useMemo(() => getJobs("ingest", listId), [getJobs, listId]);
  const [refreshedJobs, setRefreshedJobs] = useState<string[]>([]);

  const [currentSources, setCurrentSources] = useState<Source[]>(sources);
  // Sync currentSources when sources prop changes (e.g., after initial load)
  useEffect(() => {
    setCurrentSources(sources);
  }, [sources]);

  // When a job flips to done, refresh sources and dismiss
  useEffect(() => {
    const completed = ingestJobs.filter(
      (j) => j.isTerminal && j.job.status === "done" && !refreshedJobs.includes(j.job.id),
    );
    if (completed.length === 0) return;

    let cancelled = false;
    async function refresh() {
      try {
        const next = await api.listSources(listId);
        if (cancelled) return;
        setCurrentSources(next ?? []);
        for (const { job } of completed) {
          setRefreshedJobs((prev) => [...prev, job.id]);
          await dismissJob(job.id);
        }
      } catch {
        // Non-critical: leave jobs in terminal state
      }
    }
    void refresh();
    return () => {
      cancelled = true;
    };
  }, [ingestJobs, listId, refreshedJobs]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) return;
    setUrl("");
    setError(null);
    try {
      // TODO: F3 — replace with preview-then-ingest flow
      const result = await api.ingestUrl(listId, []);
      if (!result) return;
      trackJobs(
        result.queued.map((id) => ({
          id,
          producer: "ingest",
          label: trimmed,
          contextKey: listId,
        })),
      );
    } catch (err) {
      setUrl(trimmed);
      setError(toErrorMessageWithT(err, t));
    }
  }, [listId, t, trackJobs, url]);

  const handleDelete = useCallback(async (source: Source) => {
    await api.deleteSource(listId, source.id);
    setCurrentSources((prev) => prev.filter((s) => s.id !== source.id));
  }, [listId]);

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 px-4 pt-4 pb-3">
        <form className="relative" onSubmit={handleSubmit}>
          <input
            type="text"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              setError(null);
            }}
            placeholder={t("lists.pasteUrl")}
            className="w-full rounded-full border border-border bg-white/80 py-2.5 pr-10 pl-4 text-sm text-ink placeholder:text-muted/50 outline-none focus:border-blue/40 focus:bg-white transition"
          />
          <button
            type="submit"
            disabled={!url.trim()}
            aria-label="Add source"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 flex h-7 w-7 items-center justify-center rounded-full text-muted transition disabled:opacity-0 enabled:hover:bg-blue enabled:hover:text-white enabled:hover:shadow-sm"
          >
            <ArrowRight size={15} />
          </button>
        </form>
        {error && <p className="mt-1.5 px-4 text-xs text-pink">{error}</p>}
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto px-4 pb-4">
        {ingestJobs
          .filter((j) => !j.isTerminal || j.job.status === "failed" || j.job.status === "needs_auth")
          .map((item) => (
            <IngestingSourceRow
              key={item.job.id}
              stage={item.job.status}
              title={item.label}
              onDismiss={() => void dismissJob(item.job.id)}
              t={t}
            />
          ))}
        {currentSources.map((source) => (
          <SourceRow
            key={source.id}
            source={source}
            onOpen={() => onOpenSource(source)}
            onDelete={() => handleDelete(source)}
            t={t}
          />
        ))}
      </div>
    </div>
  );
}
