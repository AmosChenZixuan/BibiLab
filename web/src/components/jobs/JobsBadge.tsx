import { useEffect, useMemo, useState } from "react";

import { api, toErrorMessage } from "../../lib/api";
import type { Job } from "../../lib/types";

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
    const intervalId = window.setInterval(() => {
      void loadJobs();
    }, 5000);

    return () => {
      cancelled = true;
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
        className="jobs-pill"
        aria-label={formatActiveJobsLabel(activeJobs.length)}
        aria-expanded={isOpen}
        aria-controls="jobs-drawer"
        onClick={() => setIsOpen((open) => !open)}
      >
        <strong>Jobs</strong>
        <span className="jobs-pill__count">{formatActiveJobsLabel(activeJobs.length)}</span>
      </button>
      {isOpen ? (
        <>
          <button
            type="button"
            className="jobs-drawer-backdrop"
            aria-label="Close jobs drawer"
            onClick={() => setIsOpen(false)}
          />
          <section id="jobs-drawer" className="jobs-drawer" aria-label="Jobs">
            <div className="jobs-drawer__header">
              <div>
                <h2 className="jobs-drawer__title">Jobs</h2>
                <p className="muted">Background ingestion and model work.</p>
              </div>
              <button type="button" className="ghost-button" onClick={() => setIsOpen(false)}>
                Close
              </button>
            </div>
            {errorMessage ? <p className="status-message error">{errorMessage}</p> : null}
            <div className="jobs-list">
              {jobs.length === 0 ? (
                <div className="jobs-row jobs-row--empty">
                  <p className="muted">No jobs yet.</p>
                </div>
              ) : (
                jobs.map((job) => {
                  const jobTitle = getJobTitle(job);
                  const isTerminal = TERMINAL_STATUSES.has(job.status);

                  return (
                    <article key={job.id} className="jobs-row">
                      <div className="jobs-row__summary">
                        <div className="row">
                          <h3 className="jobs-row__title">{jobTitle}</h3>
                          <span className={`status-chip ${job.status === "failed" ? "error" : job.status === "done" ? "ok" : "unavailable"}`}>
                            {job.status}
                          </span>
                        </div>
                        <p className="muted">{job.progress}%</p>
                        {job.error ? <p className="status-message error">{job.error}</p> : null}
                      </div>
                      {!isTerminal ? (
                        <button
                          type="button"
                          className="ghost-button"
                          aria-label={`Cancel ${jobTitle}`}
                          disabled={cancellingJobId === job.id}
                          onClick={() => void handleCancel(job.id)}
                        >
                          {cancellingJobId === job.id ? "Cancelling..." : "Cancel"}
                        </button>
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
