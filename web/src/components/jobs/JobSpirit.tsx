import { X } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { ARTIFACT_TYPE_KEYS } from "@/lib/artifact-types";
import type { JobActivityItem } from "./JobActivityProvider";
import { getJobTitle, getJobTone, useJobActivity } from "./JobActivityProvider";

function StatusDot({ item }: { item: JobActivityItem }) {
  if (!item.isTerminal) {
    return (
      <span
        className="inline-block h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-blue"
        aria-hidden="true"
      />
    );
  }
  const tone = getJobTone(item.job);
  const color =
    tone === "ok" ? "bg-blue" : tone === "error" ? "bg-pink" : "bg-muted";
  return (
    <span className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${color}`} aria-hidden="true" />
  );
}

export function JobSpirit() {
  const { t } = useLanguage();
  const {
    activeJobs,
    isJobPending,
    clearTerminalJobs,
    dismissJob,
    cancelJob,
    errorMessage,
    isPanelOpen,
    setPanelOpen,
    visibleJobs,
  } = useJobActivity();

  if (visibleJobs.length === 0) {
    return null;
  }

  const hasTerminal = visibleJobs.some((item) => item.isTerminal);

  return (
    <>
      {isPanelOpen ? (
        <button
          type="button"
          className="fixed inset-0 z-float border-0 bg-transparent"
          aria-label={t("common.close")}
          onClick={() => setPanelOpen(false)}
        />
      ) : null}
      <div className="fixed bottom-4 right-4 z-float flex flex-col items-end gap-2 max-md:bottom-3 max-md:right-3">
        {isPanelOpen ? (
          <section
            className="w-80 overflow-hidden rounded-3xl border border-border bg-white/96 shadow-lg backdrop-blur-md"
            aria-label={t("jobs.jobs")}
          >
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <span className="text-sm font-semibold text-ink">
                {activeJobs.length > 0 ? t("jobs.running", { count: activeJobs.length }) : t("jobs.jobs")}
              </span>
              {hasTerminal ? (
                <button
                  type="button"
                  className="text-xs text-muted transition-colors hover:text-ink"
                  onClick={clearTerminalJobs}
                >
                  {t("jobs.clearDone")}
                </button>
              ) : null}
            </div>
            {errorMessage ? (
              <p className="px-4 pt-3 text-xs text-pink">{errorMessage}</p>
            ) : null}
            <ul className="max-h-96 overflow-y-auto divide-y divide-border">
              {visibleJobs.map((item) => {
                const fallbackLabel = item.producer === "artifact"
                  ? (ARTIFACT_TYPE_KEYS[item.label] ? t(ARTIFACT_TYPE_KEYS[item.label]) : item.label)
                  : item.label;
                const title = getJobTitle(item.job, fallbackLabel);
                return (
                  <li
                    key={item.job.id}
                    className={`min-w-0 grid gap-2 px-4 py-3 transition-opacity ${item.isTerminal ? "opacity-50" : ""}`}
                  >
                    <div className="flex items-center gap-2.5 overflow-hidden">
                      <StatusDot item={item} />
                      <span
                        className="min-w-0 flex-1 truncate text-sm font-medium text-ink"
                        title={title}
                      >
                        {title}
                      </span>
                      {(item.isTerminal || item.producer === "ingest") ? (
                        <button
                          type="button"
                          className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-muted transition-colors hover:bg-border hover:text-ink disabled:opacity-30"
                          aria-label={item.isTerminal ? `Dismiss ${title}` : `Cancel ${title}`}
                          disabled={isJobPending(item.job.id)}
                          onClick={() =>
                            item.isTerminal
                              ? void dismissJob(item.job.id)
                              : void cancelJob(item.job.id)
                          }
                        >
                          <X size={12} />
                        </button>
                      ) : null}
                    </div>
                    {!item.isTerminal ? (
                      <div className="h-px overflow-hidden rounded-full bg-border">
                        <div
                          className="h-full bg-blue transition-all duration-700"
                          style={{ width: `${Math.max(item.job.progress, 4)}%` }}
                        />
                      </div>
                    ) : null}
                    {item.job.error ? (
                      <p className="text-xs text-pink break-words min-w-0">{item.job.error}</p>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          </section>
        ) : null}

        <button
          type="button"
          aria-label={t("jobs.jobs")}
          className="inline-flex items-center gap-2 rounded-full border border-border bg-white/94 px-3 py-1.5 text-sm text-ink shadow-lg backdrop-blur-md transition-shadow hover:shadow-lg"
          onClick={() => setPanelOpen(!isPanelOpen)}
        >
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full bg-blue transition-opacity ${activeJobs.length > 0 ? "animate-pulse" : "opacity-40"}`}
            aria-hidden="true"
          />
          <span className="font-medium">
            {activeJobs.length > 0 ? activeJobs.length : visibleJobs.length}
          </span>
        </button>
      </div>
    </>
  );
}
