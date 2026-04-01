import { useState } from "react";

import type { WhisperModel } from "../../lib/types";

type Props = {
  models: WhisperModel[];
  onDownload: (modelSize: string) => Promise<void>;
};

export function WhisperModelsCard({ models, onDownload }: Props) {
  const [downloading, setDownloading] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  async function handleDownload(modelName: string) {
    setDownloading(modelName);
    setStatus(null);
    try {
      await onDownload(modelName);
      setStatus("Queued Whisper model download");
    } finally {
      setDownloading(null);
    }
  }

  return (
    <section>
      <div className="row">
        <div>
          <h2 className="list-card__title">Whisper models</h2>
          <p className="page-lede">Inspect what is installed and queue missing downloads.</p>
        </div>
      </div>
      <div className="model-list">
        {models.map((model) => (
          <div className="model-row" key={model.name}>
            <div className="row">
              <strong>{model.name}</strong>
              {model.selected ? <span className="status-chip ok">selected</span> : null}
              <span className={`status-chip ${model.installed ? "ok" : "error"}`}>
                {model.installed ? "installed" : "missing"}
              </span>
            </div>
            <div className="inline-actions">
              <button
                className="secondary-button"
                disabled={model.installed || downloading === model.name}
                onClick={() => void handleDownload(model.name)}
                type="button"
              >
                {downloading === model.name ? "Queueing..." : `Download ${model.name}`}
              </button>
              {model.path ? <span className="muted">{model.path}</span> : null}
            </div>
          </div>
        ))}
      </div>
      {status ? <p className="status-message success">{status}</p> : null}
    </section>
  );
}
