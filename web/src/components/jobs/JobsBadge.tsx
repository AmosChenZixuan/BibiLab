import { useEffect, useMemo, useState } from "react";

import { JOBS_REFRESH_EVENT, api, toErrorMessage } from "../../lib/api";
import type { Job } from "../../lib/types";
import { Button, StatusChip } from "../../components/ui";

const TERMINAL_STATUSES = new Set(["done", "failed"]);

function getJobTitle(job: Job): string {
  const title = job.meta.title;
  return typeof title === "string" && title.trim() ? title : job.source_url;
}

function formatActiveJobsLabel(count: number): string {
  if (count === 0) {
    return "No active jobs";
  }
  if (count === 1) {
    return "1 active job";
  }
  return `${count} active jobs`;
}

export function JobsBadge() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [cancellingJobId, setCancellingJobId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadJobs() {
      try {
        const nextJobs = await api.listJobs();
        if (!cancelled) {
          setJobs(nextJobs);
          setErrorMessage(null);
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(toErrorMessage(error));
        }
      }
    }

    void loadJobs();
    function handleRefresh() {
      void loadJobs();
    }

    window.addEventListener(JOBS_REFRESH_EVENT, handleRefresh);
    const intervalId = window.setInterval(() => {
      void loadJobs();
    }, 5000);

    return () => {
      cancelled = true;
      window.removeEventListener(JOBS_REFRESH_EVENT, handleRefresh);
      window.clearInterval(intervalId);
    };
  }, []);

  const activeJobs = useMemo(
    () => jobs.filter((job) => !TERMINAL_STATUSES.has(job.status)),
    [jobs],
  );

  async function handleCancel(jobId: string) {
    setCancellingJobId(jobId);

    try {
      await api.deleteJob(jobId);
      const nextJobs = await api.listJobs();
      setJobs(nextJobs);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(toErrorMessage(error));
    } finally {
      setCancellingJobId(null);
    }
  }

  return (
    <>
      <button
        type="button"
        className="inline-flex items-center gap-2 rounded-full border border-border bg-white/76 px-3.5 py-2.5 text-muted/80"
        aria-label={formatActiveJobsLabel(activeJobs.length)}
        aria-expanded={isOpen}
        aria-controls="jobs-drawer"
        onClick={() => setIsOpen((open) => !open)}
      >
        <strong>Jobs</strong>
        <span className="text-muted">{formatActiveJobsLabel(activeJobs.length)}</span>
      </button>
      {isOpen ? (
        <>
          <button
            type="button"
            className="fixed inset-0 z-[19] border-0 bg-blue/14"
            aria-label="Close jobs drawer"
            onClick={() => setIsOpen(false)}
          />
          <section
            id="jobs-drawer"
            className="fixed top-[92px] /* 52px navbar + 40px gap */ right-6 z-20 grid max-h-[calc(100vh-116px)] /* keeps drawer within viewport */ w-[min(420px,calc(100vw-24px))] /* drawer max-width: 420px or full viewport */ gap-4 overflow-auto rounded-drawer border border-border bg-white/92 p-5 shadow-elevated backdrop-blur-[18px] /* glass blur — no matching Tailwind step */ max-[820px]:top-[84px] max-[820px]:right-3 max-[820px]:w-[calc(100vw-24px)]"
            aria-label="Jobs"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="m-0 font-serif">Jobs</h2>
                <p className="m-0 text-muted">Background ingestion and model work.</p>
              </div>
              <Button type="button" variant="ghost" onClick={() => setIsOpen(false)}>
                Close
              </Button>
            </div>
            {errorMessage ? <p className="m-0 text-sm text-danger">{errorMessage}</p> : null}
            <div className="grid gap-3">
              {jobs.length === 0 ? (
                <div className="grid min-h-[120px] place-items-center rounded-3xl border border-border bg-[rgba(248,251,255,0.72)] /* sky-tinted card background */ p-4">
                  <p className="m-0 text-muted">No jobs yet.</p>
                </div>
              ) : (
                jobs.map((job) => {
                  const jobTitle = getJobTitle(job);
                  const isTerminal = TERMINAL_STATUSES.has(job.status);

                  return (
                    <article
                      key={job.id}
                      className="grid gap-3.5 rounded-3xl border border-border bg-[rgba(248,251,255,0.72)] /* sky-tinted card background */ p-4"
                    >
                      <div className="grid gap-2">
                        <div className="flex flex-wrap items-center gap-3">
                          <h3 className="m-0 font-serif">{jobTitle}</h3>
                          <StatusChip status={job.status === "failed" ? "error" : job.status === "done" ? "ok" : "unavailable"}>
                            {job.status}
                          </StatusChip>
                        </div>
                        <p className="m-0 text-muted">{job.progress}%</p>
                        {job.error ? <p className="m-0 text-sm text-danger">{job.error}</p> : null}
                      </div>
                      {!isTerminal ? (
                        <Button
                          type="button"
                          variant="ghost"
                          aria-label={`Cancel ${jobTitle}`}
                          disabled={cancellingJobId === job.id}
                          onClick={() => void handleCancel(job.id)}
                        >
                          {cancellingJobId === job.id ? "Cancelling..." : "Cancel"}
                        </Button>
                      ) : null}
                    </article>
                  );
                })
              )}
            </div>
          </section>
        </>
      ) : null}
    </>
  );
}
