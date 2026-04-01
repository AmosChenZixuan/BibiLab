import { useState } from "react";

import type { Source } from "../../lib/types";

type Props = {
  busy: boolean;
  error: string | null;
  ingestStatus: string | null;
  onDelete: (source: Source) => Promise<void>;
  onIngest: (url: string) => Promise<void>;
  onOpen: (source: Source) => void;
  sources: Source[];
};

export function SourceList({ busy, error, ingestStatus, onDelete, onIngest, onOpen, sources }: Props) {
  const [url, setUrl] = useState("");

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!url.trim()) {
      return;
    }
    await onIngest(url.trim());
    setUrl("");
  }

  return (
    <div className="workspace-panel__body">
      <form className="form-stack" onSubmit={handleSubmit}>
        <label className="field">
          <span>Source URL</span>
          <input
            aria-label="Source URL"
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://www.bilibili.com/video/..."
            value={url}
          />
        </label>
        <div className="inline-actions">
          <button className="primary-button" disabled={busy} type="submit">
            {busy ? "Queueing..." : "Queue source"}
          </button>
          {ingestStatus ? <p className="status-message success">{ingestStatus}</p> : null}
        </div>
        {error ? <p className="status-message error">{error}</p> : null}
      </form>

      <div className="source-list">
        {sources.length === 0 ? (
          <div className="source-row source-row--empty">
            <p className="page-lede">No sources yet. Queue a Bilibili URL to start building this notebook.</p>
          </div>
        ) : (
          sources.map((source) => (
            <article className="source-row" key={source.video_id}>
              <button
                aria-label={`Open ${source.title}`}
                className="source-row__open"
                onClick={() => onOpen(source)}
                type="button"
              >
                <strong>{source.title}</strong>
                <span className="page-lede">{source.platform}</span>
              </button>
              <button
                aria-label={`Delete ${source.title}`}
                className="ghost-button"
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
