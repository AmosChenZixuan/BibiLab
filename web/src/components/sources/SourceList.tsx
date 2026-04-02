import { useState } from "react";

import type { Source } from "../../lib/types";
import {
  checkboxRowClass,
  fieldClass,
  fieldLabelClass,
  ghostButtonClass,
  inputClass,
  mutedTextClass,
  primaryButtonClass,
  statusErrorClass,
  statusSuccessClass,
  workspacePanelBodyClass,
} from "../../lib/ui";

type Props = {
  busy: boolean;
  error: string | null;
  ingestStatus: string | null;
  onDelete: (source: Source) => Promise<void>;
  onIngest: (url: string, rerun: boolean) => Promise<void>;
  onOpen: (source: Source) => void;
  sources: Source[];
};

export function SourceList({ busy, error, ingestStatus, onDelete, onIngest, onOpen, sources }: Props) {
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
    <div className={workspacePanelBodyClass}>
      <form className="grid gap-4" onSubmit={handleSubmit}>
        <label className={fieldClass}>
          <span className={fieldLabelClass}>Source URL</span>
          <input
            aria-label="Source URL"
            className={inputClass}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://www.bilibili.com/video/..."
            value={url}
          />
        </label>
        <label className={checkboxRowClass}>
          <input
            aria-label="Re-run existing source"
            checked={rerun}
            onChange={(event) => setRerun(event.target.checked)}
            type="checkbox"
          />
          <span>Re-run existing source</span>
        </label>
        <div className="flex flex-wrap items-center gap-3">
          <button className={primaryButtonClass} disabled={busy} type="submit">
            {busy ? "Queueing..." : "Queue source"}
          </button>
          {ingestStatus ? <p className={statusSuccessClass}>{ingestStatus}</p> : null}
        </div>
        {error ? <p className={statusErrorClass}>{error}</p> : null}
      </form>

      <div className="grid gap-3">
        {sources.length === 0 ? (
          <div className="flex justify-start rounded-2xl border border-border bg-white/64 px-4 py-3.5">
            <p className={mutedTextClass}>No sources yet. Queue a Bilibili URL to start building this notebook.</p>
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
                <span className={mutedTextClass}>{source.platform}</span>
              </button>
              <button
                aria-label={`Delete ${source.title}`}
                className={ghostButtonClass}
                onClick={() => void onDelete(source)}
                type="button"
              >
                Delete
              </button>
            </article>
          ))
        )}
      </div>
    </div>
  );
}
