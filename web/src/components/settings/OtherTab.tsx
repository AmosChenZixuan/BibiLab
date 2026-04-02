import { useEffect, useId, useState } from "react";

import type { HealthDependency, LocusConfig } from "../../lib/types";
import {
  fieldHintClass,
  fieldLabelClass,
  settingsControlClass,
  settingsFieldClass,
  settingsFieldMetaClass,
  settingsInputClass,
  statusChipClass,
} from "../../lib/ui";

type OtherTabProps = {
  config: LocusConfig;
  dependencies: Record<string, HealthDependency>;
  onBlur: (updated: LocusConfig) => void;
};

export function OtherTab({ config, dependencies, onBlur }: OtherTabProps) {
  const [local, setLocal] = useState({
    workerConcurrency: config.backend.worker_concurrency,
  });

  useEffect(() => {
    setLocal({
      workerConcurrency: config.backend.worker_concurrency,
    });
  }, [config]);

  function handleBlur() {
    onBlur({
      ...config,
      backend: {
        ...config.backend,
        worker_concurrency: local.workerConcurrency,
      },
    });
  }

  const backendUrl = `http://localhost:${config.backend.port}`;
  const embeddingDependency = dependencies.embedding_model;
  const backendDependency = dependencies.backend;
  const ffmpegDependency = dependencies.ffmpeg;
  const workerConcurrencyId = useId();

  const embeddingPath = embeddingDependency?.status === "ok" ? embeddingDependency.message : null;
  const ffmpegPath = ffmpegDependency?.status === "ok" ? ffmpegDependency.message : null;

  const statusRowClass = `${settingsFieldClass} items-center`;
  const valueClass = "ml-auto text-right font-mono text-[0.82rem] text-[#8096b3]";

  return (
    <div className="grid gap-4">
      <div className={statusRowClass}>
        <div className={settingsFieldMetaClass}>
          <div className="flex flex-wrap items-center gap-2">
            <p className={fieldLabelClass}>Embedding Model</p>
            <span className={statusChipClass(embeddingDependency?.status === "ok" ? "ok" : "error")}>
              {embeddingDependency?.status === "ok" ? "ready" : "missing"}
            </span>
          </div>
          <p className={fieldHintClass}>If missing, the first processing run downloads embeddings before indexing, which makes startup slower.</p>
        </div>
        <div className="flex min-w-[220px] flex-1 items-center justify-end gap-3 self-center">
          {embeddingDependency?.status === "ok" ? (
            <p className={valueClass}>
              {embeddingPath}
            </p>
          ) : (
            <span className="max-w-[38rem] text-right text-[0.88rem] leading-6 text-[#5b7faa]">
              {embeddingDependency?.message}
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-x-5 gap-y-2 bg-[rgba(255,255,255,0.36)] px-4 py-3 md:grid-cols-[minmax(0,1fr)_320px]">
        <div className="flex flex-wrap items-center gap-2">
          <p className={fieldLabelClass}>Backend API</p>
          <span className={statusChipClass(backendDependency?.status === "ok" ? "ok" : "error")}>
            {backendDependency?.status === "ok" ? "connected" : "offline"}
          </span>
        </div>
        <div className="flex items-center justify-end">
          <p className={valueClass}>{backendUrl}</p>
        </div>

        <p className={fieldHintClass}>Required. If the backend is offline, the web app cannot load or save configuration.</p>
        <div />

        <div className="border-l border-[rgba(106,147,198,0.18)] pl-4">
          <div className="grid gap-2">
            <label className={fieldLabelClass} htmlFor={workerConcurrencyId}>Worker Concurrency</label>
            <p className={fieldHintClass}>Controls parallel jobs. Higher values improve throughput but increase local resource usage.</p>
          </div>
        </div>
        <div className="flex items-center justify-end">
          <input
            aria-label="Worker Concurrency"
            className={`${settingsInputClass} ${settingsControlClass}`}
            id={workerConcurrencyId}
            max={8}
            min={1}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocal((current) => ({
                ...current,
                workerConcurrency: Number(event.target.value),
              }))
            }
            type="number"
            value={local.workerConcurrency}
          />
        </div>
      </div>

      <div className={statusRowClass}>
        <div className={settingsFieldMetaClass}>
          <div className="flex flex-wrap items-center gap-2">
            <p className={fieldLabelClass}>FFmpeg</p>
            <span
              className={statusChipClass(ffmpegDependency?.status === "ok" ? "ok" : "error")}
              title={ffmpegDependency?.status === "ok" ? "FFmpeg installed" : "FFmpeg not found"}
            >
              {ffmpegDependency?.status === "ok" ? "installed" : "missing"}
            </span>
          </div>
          <p className={fieldHintClass}>Required. Without FFmpeg, media audio cannot be extracted, so ingestion fails.</p>
        </div>
        <div className="flex min-w-[220px] flex-1 items-center justify-end gap-3 self-center">
          {ffmpegDependency?.status === "ok" ? (
            <p className={valueClass}>
              {ffmpegPath}
            </p>
          ) : (
            <a
              className="text-[#5b7faa] underline"
              href="https://ffmpeg.org/download.html"
              rel="noreferrer"
              target="_blank"
            >
              Download FFmpeg
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
