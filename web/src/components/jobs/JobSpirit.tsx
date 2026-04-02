import { MdClose } from "react-icons/md";

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
    tone === "ok" ? "bg-success" : tone === "error" ? "bg-danger" : "bg-muted";
  return (
    <span className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${color}`} aria-hidden="true" />
  );
}

export function JobSpirit() {
  const {
    activeJobs,
    cancellingJobId,
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
          className="fixed inset-0 z-[189] border-0 bg-transparent"
          aria-label="Close background jobs"
          onClick={() => setPanelOpen(false)}
        />
      ) : null}
      <div className="fixed right-4 bottom-4 z-[190] flex flex-col items-end gap-2 max-[820px]:right-3 max-[820px]:bottom-3">
        {isPanelOpen ? (
          <section
            className="w-[min(340px,calc(100vw-24px))] overflow-hidden rounded-drawer border border-border bg-white/96 shadow-elevated backdrop-blur-[18px]"
            aria-label="Background jobs"
          >
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <span className="text-sm font-semibold text-ink">
                {activeJobs.length > 0 ? `${activeJobs.length} running` : "Jobs"}
              </span>
              {hasTerminal ? (
                <button
                  type="button"
                  className="text-xs text-muted transition-colors hover:text-ink"
                  onClick={clearTerminalJobs}
                >
                  Clear done
                </button>
              ) : null}
            </div>
            {errorMessage ? (
              <p className="px-4 pt-3 text-xs text-danger">{errorMessage}</p>
            ) : null}
            <ul className="divide-y divide-border">
              {visibleJobs.map((item) => {
                const title = getJobTitle(item.job, item.label);
                return (
                  <li
                    key={item.job.id}
                    className={`grid gap-2 px-4 py-3 transition-opacity ${item.isTerminal ? "opacity-50" : ""}`}
                  >
                    <div className="flex items-center gap-2.5">
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
                          disabled={cancellingJobId === item.job.id}
                          onClick={() =>
                            item.isTerminal
                              ? dismissJob(item.job.id)
                              : void cancelJob(item.job.id)
                          }
                        >
                          <MdClose size={12} />
                        </button>
                      ) : null}
                    </div>
                    {!item.isTerminal ? (
                      <div className="h-px overflow-hidden rounded-full bg-border">
                        <div
                          className="h-full bg-blue/50 transition-[width] duration-700"
                          style={{ width: `${Math.max(item.job.progress, 4)}%` }}
                        />
                      </div>
                    ) : null}
                    {item.job.error ? (
                      <p className="text-xs text-danger">{item.job.error}</p>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          </section>
        ) : null}

        <button
          type="button"
          aria-label="Background jobs"
          className="inline-flex items-center gap-2 rounded-full border border-border bg-white/94 px-3 py-1.5 text-sm text-ink shadow-card backdrop-blur-[18px] transition-shadow hover:shadow-elevated"
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
