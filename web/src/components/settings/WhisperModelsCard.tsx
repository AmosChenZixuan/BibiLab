import { useState } from "react";

import type { WhisperModel } from "../../lib/types";
import {
  appPanelClass,
  mutedTextClass,
  secondaryButtonClass,
  sectionTitleClass,
  statusChipClass,
  statusSuccessClass,
} from "../../lib/ui";

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
    <section className={`${appPanelClass} grid gap-4`}>
      <div className="flex flex-wrap items-center gap-3">
        <div>
          <h2 className={sectionTitleClass}>Whisper models</h2>
          <p className={mutedTextClass}>Inspect what is installed and queue missing downloads.</p>
        </div>
      </div>
      <div className="grid gap-3">
        {models.map((model) => (
          <div className="grid gap-2 border-t border-[rgba(106,147,198,0.12)] pt-3" key={model.name}>
            <div className="flex flex-wrap items-center gap-3">
              <strong>{model.name}</strong>
              {model.selected ? <span className={statusChipClass("ok")}>selected</span> : null}
              <span className={statusChipClass(model.installed ? "ok" : "error")}>
                {model.installed ? "installed" : "missing"}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button
                className={secondaryButtonClass}
                disabled={model.installed || downloading === model.name}
                onClick={() => void handleDownload(model.name)}
                type="button"
              >
                {downloading === model.name ? "Queueing..." : `Download ${model.name}`}
              </button>
              {model.path ? <span className={mutedTextClass}>{model.path}</span> : null}
            </div>
          </div>
        ))}
      </div>
      {status ? <p className={statusSuccessClass}>{status}</p> : null}
    </section>
  );
}
