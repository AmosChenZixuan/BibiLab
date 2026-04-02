import { useState } from "react";

import { MdClose } from "react-icons/md";

import type { Source } from "../../lib/types";
import { Button, FormField, Input, PanelBody } from "../../components/ui";
import type { JobActivityItem } from "../jobs/JobActivityProvider";
import { getJobTitle, getJobTone } from "../jobs/JobActivityProvider";

type Props = {
  busy: boolean;
  error: string | null;
  ingestStatus: string | null;
  ingestJobs: JobActivityItem[];
  onDismissIngestJob: (jobId: string) => void;
  onDelete: (source: Source) => Promise<void>;
  onIngest: (url: string, rerun: boolean) => Promise<void>;
  onOpen: (source: Source) => void;
  sources: Source[];
};

export function SourceList({
  busy,
  error,
  ingestStatus,
  ingestJobs,
  onDelete,
  onDismissIngestJob,
  onIngest,
  onOpen,
  sources,
}: Props) {
  const [url, setUrl] = useState("");
  const [rerun, setRerun] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!url.trim()) {
      return;
    }
    await onIngest(url.trim(), rerun);
    setUrl("");
  }

  return (
    <PanelBody>
      <form className="grid gap-4" onSubmit={handleSubmit}>
        <FormField label="Source URL">
          <Input
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://www.bilibili.com/video/..."
            value={url}
          />
        </FormField>
        <label className="inline-flex items-center gap-2.5">
          <input
            aria-label="Re-run existing source"
            checked={rerun}
            onChange={(event) => setRerun(event.target.checked)}
            type="checkbox"
          />
          <span>Re-run existing source</span>
        </label>
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="primary" disabled={busy} type="submit">
            {busy ? "Queueing..." : "Queue source"}
          </Button>
          {ingestStatus ? <p className="m-0 text-sm text-success">{ingestStatus}</p> : null}
        </div>
        {error ? <p className="m-0 text-sm text-danger">{error}</p> : null}
      </form>

      {ingestJobs.length > 0 ? (
        <ul className="divide-y divide-border overflow-hidden rounded-2xl border border-border bg-white/64">
          {ingestJobs.map((item) => {
            const title = getJobTitle(item.job, item.label);
            const tone = getJobTone(item.job);
            const accentColor = !item.isTerminal
              ? "border-l-blue/40"
              : tone === "ok"
                ? "border-l-success/50"
                : "border-l-danger/50";

            return (
              <li
                key={item.job.id}
                className={`grid border-l-2 transition-opacity ${accentColor} ${item.isTerminal ? "opacity-50" : ""}`}
              >
                <div className="flex items-center gap-3 px-4 py-3">
                  <span className="min-w-0 flex-1 truncate text-sm font-medium text-ink">
                    {title}
                  </span>
                  {item.isTerminal ? (
                    <button
                      type="button"
                      className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-muted transition-colors hover:bg-border hover:text-ink"
                      aria-label={`Dismiss ${title}`}
                      onClick={() => onDismissIngestJob(item.job.id)}
                    >
                      <MdClose size={12} />
                    </button>
                  ) : null}
                </div>
                {!item.isTerminal ? (
                  <div className="h-px overflow-hidden bg-border">
                    <div
                      className="h-full bg-blue/50 transition-[width] duration-700"
                      style={{ width: `${Math.max(item.job.progress, 4)}%` }}
                    />
                  </div>
                ) : null}
                {item.job.error ? (
                  <p className="px-4 pb-3 text-xs text-danger">{item.job.error}</p>
                ) : null}
              </li>
            );
          })}
        </ul>
      ) : null}

      <div className="grid gap-3">
        {sources.length === 0 ? (
          <div className="flex justify-start rounded-2xl border border-border bg-white/64 px-4 py-3.5">
            <p className="m-0 text-muted">No sources yet. Queue a Bilibili URL to start building this notebook.</p>
          </div>
        ) : (
          sources.map((source) => (
            <article
              className="flex items-center justify-between gap-3 rounded-2xl border border-border bg-white/64 px-4 py-3.5"
              key={source.video_id}
            >
              <button
                aria-label={`Open ${source.title}`}
                className="grid gap-1 border-0 bg-transparent text-left text-inherit"
                onClick={() => onOpen(source)}
                type="button"
              >
                <strong>{source.title}</strong>
                <span className="m-0 text-muted">{source.platform}</span>
              </button>
              <Button
                aria-label={`Delete ${source.title}`}
                variant="ghost"
                onClick={() => void onDelete(source)}
                type="button"
              >
                Delete
              </Button>
            </article>
          ))
        )}
      </div>
    </PanelBody>
  );
}
