import { useEffect, useId, useState } from "react";

import { useJobActivity } from "../jobs/JobActivityProvider";
import type { JobActivityItem } from "../jobs/JobActivityProvider";
import { api } from "../../lib/api";
import type { HealthDependency, LocusConfig, WhisperModel } from "../../lib/types";
import { MdDownload } from "react-icons/md";

import { Select, SettingsField, Spinner } from "../../components/ui";

type ModelDownloadCellProps = {
  modelName: string;
  modelJob: JobActivityItem | null;
  downloading: string | null;
  onDownload: (name: string) => Promise<void>;
};

function ModelDownloadCell({ modelName, modelJob, downloading, onDownload }: ModelDownloadCellProps) {
  if (modelJob && !modelJob.isTerminal) {
    return (
      <Spinner label={`Downloading ${modelName}`} />
    );
  }

  if (modelJob?.job.status === "failed" || modelJob?.job.status === "needs_auth") {
    return <span className="text-xs text-rose-900">{modelJob.job.error ?? "Failed"}</span>;
  }

  return (
    <button
      type="button"
      className="flex items-center justify-center text-sky"
      aria-label={`Download ${modelName}`}
      disabled={downloading === modelName}
      onClick={() => void onDownload(modelName)}
    >
      <MdDownload size={18} />
    </button>
  );
}

type TranscriptTabProps = {
  config: LocusConfig;
  dependencies: Record<string, HealthDependency>;
  onBlur: (updated: LocusConfig) => void;
};

export function TranscriptTab({ config, dependencies, onBlur }: TranscriptTabProps) {
  const { dismissJob, getJobs, trackJobs } = useJobActivity();
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

  const modelJobs = getJobs("whisper_download");
  const terminalModelJobCount = modelJobs.filter((j) => j.isTerminal).length;

  useEffect(() => {
    if (terminalModelJobCount === 0) {
      return;
    }

    const terminalIds = modelJobs.filter((j) => j.isTerminal).map((j) => j.job.id);

    async function refreshAndDismiss() {
      await refreshModels();
      for (const id of terminalIds) {
        dismissJob(id);
      }
    }

    void refreshAndDismiss();
  }, [terminalModelJobCount]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleBlur() {
    onBlur({ ...config, transcription: localTranscription });
  }

  async function handleDownload(modelName: string) {
    setDownloading(modelName);
    try {
      const response = await api.downloadWhisperModel(modelName);
      trackJobs([
        {
          id: response.job_id,
          producer: "whisper_download",
          label: response.model_size,
          contextKey: response.model_size,
        },
      ]);
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
        <SettingsField
          label="Model Size"
          hint="Required. Missing the Whisper model blocks transcript generation entirely."
          htmlFor={modelSizeId}
        >
          <Select
            aria-label="Model Size"
            id={modelSizeId}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalTranscription((current) => ({ ...current, model_size: event.target.value }))
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
          </Select>
        </SettingsField>

        <SettingsField label="Device" hint={deviceHint} htmlFor={deviceId}>
          <Select
            aria-label="Device"
            id={deviceId}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalTranscription((current) => ({ ...current, device: event.target.value }))
            }
            value={localTranscription.device}
          >
            <option value="cpu">CPU</option>
            <option value="cuda" disabled={!cudaAvailable}>
              CUDA
            </option>
          </Select>
        </SettingsField>

        <SettingsField
          label="Language"
          hint="Controls decoder accuracy. A wrong language setting can reduce transcript quality."
          htmlFor={languageId}
        >
          <Select
            aria-label="Language"
            id={languageId}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalTranscription((current) => ({ ...current, language: event.target.value }))
            }
            value={localTranscription.language}
          >
            <option value="auto">Auto</option>
            <option value="zh">Chinese</option>
            <option value="en">English</option>
          </Select>
        </SettingsField>
      </div>

      <div className="grid gap-3">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted">
          Model Downloads
        </p>
        <p className="text-sm leading-5 text-muted">Whisper model files are required. If missing, transcription cannot start until a model is downloaded.</p>
        <div className="overflow-hidden rounded-2xl border border-border bg-white/64">
          <table className="w-full border-collapse text-left">
            <tbody>
              {models.map((model) => (
                <tr key={model.name} className="border-t border-border">
                  <td className="px-4 py-3 font-semibold text-ink">{model.name}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end">
                      {model.installed ? (
                        <p className="text-right font-mono text-sm text-muted">{model.path}</p>
                      ) : (
                        <ModelDownloadCell
                          modelName={model.name}
                          modelJob={modelJobs.find((job) => job.contextKey === model.name) ?? null}
                          downloading={downloading}
                          onDownload={handleDownload}
                        />
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
