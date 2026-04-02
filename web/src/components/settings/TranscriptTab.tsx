import { useEffect, useId, useState } from "react";

import { api } from "../../lib/api";
import type { HealthDependency, LocusConfig, WhisperModel } from "../../lib/types";
import {
  fieldHintClass,
  fieldLabelClass,
  secondaryButtonClass,
  settingsControlClass,
  settingsFieldClass,
  settingsFieldMetaClass,
  settingsInputClass,
  settingsSelectClass,
} from "../../lib/ui";

type TranscriptTabProps = {
  config: LocusConfig;
  dependencies: Record<string, HealthDependency>;
  onBlur: (updated: LocusConfig) => void;
};

export function TranscriptTab({ config, dependencies, onBlur }: TranscriptTabProps) {
  const [localTranscription, setLocalTranscription] = useState(config.transcription);
  const [models, setModels] = useState<WhisperModel[]>([]);
  const [downloading, setDownloading] = useState<string | null>(null);
  const modelSizeId = useId();
  const deviceId = useId();
  const languageId = useId();

  useEffect(() => {
    setLocalTranscription(config.transcription);
  }, [config]);

  async function refreshModels(cancelled = false) {
    try {
      const nextModels = await api.listWhisperModels();
      if (!cancelled) {
        setModels(nextModels);
      }
    } catch {
      if (!cancelled) {
        setModels([]);
      }
    }
  }

  useEffect(() => {
    let cancelled = false;

    void refreshModels(cancelled);
    return () => {
      cancelled = true;
    };
  }, []);

  function handleBlur() {
    onBlur({ ...config, transcription: localTranscription });
  }

  async function handleDownload(modelName: string) {
    setDownloading(modelName);
    try {
      await api.downloadWhisperModel(modelName);
      await refreshModels();
    } finally {
      setDownloading(null);
    }
  }

  const installedModels = models.filter((model) => model.installed);
  const hasSelectedInstalledModel = installedModels.some(
    (model) => model.name === localTranscription.model_size,
  );
  const cudaDependency = dependencies.cuda;
  const cudaAvailable = cudaDependency?.status === "ok";
  const deviceHint = cudaAvailable
    ? "CUDA speeds up transcription. Without it, transcripts still run but take longer on CPU."
    : cudaDependency?.message ?? "CUDA is unavailable, so transcription will run on CPU.";

  return (
    <div className="grid gap-4">
      <div className="flex flex-col gap-3">
        <div className={settingsFieldClass}>
          <div className={settingsFieldMetaClass}>
            <label className={fieldLabelClass} htmlFor={modelSizeId}>Model Size</label>
            <p className={fieldHintClass}>Required. Missing the Whisper model blocks transcript generation entirely.</p>
          </div>
          <select
            aria-label="Model Size"
            className={settingsSelectClass}
            id={modelSizeId}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalTranscription((current) => ({
                ...current,
                model_size: event.target.value,
              }))
            }
            value={localTranscription.model_size}
          >
            {!hasSelectedInstalledModel ? (
              <option disabled value={localTranscription.model_size}>
                {localTranscription.model_size} (download required)
              </option>
            ) : null}
            {installedModels.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>
        </div>

        <div className={settingsFieldClass}>
          <div className={settingsFieldMetaClass}>
            <label className={fieldLabelClass} htmlFor={deviceId}>Device</label>
            <p className={fieldHintClass}>{deviceHint}</p>
          </div>
          <select
            aria-label="Device"
            className={settingsSelectClass}
            id={deviceId}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalTranscription((current) => ({
                ...current,
                device: event.target.value,
              }))
            }
            value={localTranscription.device}
          >
            <option value="cpu">CPU</option>
            {cudaAvailable && <option value="cuda">CUDA</option>}
          </select>
        </div>

        <div className={settingsFieldClass}>
          <div className={settingsFieldMetaClass}>
            <label className={fieldLabelClass} htmlFor={languageId}>Language</label>
            <p className={fieldHintClass}>Controls decoder accuracy. A wrong language setting can reduce transcript quality.</p>
          </div>
          <select
            aria-label="Language"
            className={settingsSelectClass}
            id={languageId}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalTranscription((current) => ({
                ...current,
                language: event.target.value,
              }))
            }
            value={localTranscription.language}
          >
            <option value="auto">Auto</option>
            <option value="zh">Chinese</option>
            <option value="en">English</option>
          </select>
        </div>
      </div>

      <div className="grid gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted">
          Model Downloads
        </p>
        <p className={fieldHintClass}>Whisper model files are required. If missing, transcription cannot start until a model is downloaded.</p>
        <div className="overflow-hidden rounded-2xl border border-border bg-white/64">
          <table className="w-full border-collapse text-left">
            <tbody>
              {models.map((model) => (
                <tr key={model.name} className="border-t border-border">
                  <td className="px-4 py-3 font-semibold text-ink">{model.name}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end">
                      {model.installed ? (
                        <p className="text-right font-mono text-sm text-muted">
                          {model.path}
                        </p>
                      ) : (
                        <button
                          className={secondaryButtonClass}
                          aria-label={`Download ${model.name}`}
                          disabled={downloading === model.name}
                          onClick={() => void handleDownload(model.name)}
                          type="button"
                        >
                          {downloading === model.name ? "Queued..." : `Download ${model.name}`}
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
