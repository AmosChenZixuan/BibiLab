import { useEffect, useId, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import type { JobActivityItem } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import type { HealthDependency, BibilabConfig, WhisperModel } from "@/lib/types";
import { Download } from "lucide-react";

import { Select, SettingsField, Spinner } from "@/components/ui";

type ModelDownloadCellProps = {
  modelName: string;
  modelJob: JobActivityItem | null;
  downloading: string | null;
  onDownload: (name: string) => Promise<void>;
  t: (key: string, params?: Record<string, string | number>) => string;
};

function ModelDownloadCell({ modelName, modelJob, downloading, onDownload, t }: ModelDownloadCellProps) {
  if (modelJob && !modelJob.isTerminal) {
    return (
      <Spinner label={t("common.downloading", { name: modelName })} />
    );
  }

  if (modelJob?.job.status === "failed" || modelJob?.job.status === "needs_auth") {
    return <span className="text-xs text-rose-900">{modelJob.job.error ?? t("common.failed")}</span>;
  }

  return (
    <button
      type="button"
      className="flex items-center justify-center text-sky"
      aria-label="Download"
      disabled={downloading === modelName}
      onClick={() => void onDownload(modelName)}
    >
      <Download size={18} />
    </button>
  );
}

type TranscriptTabProps = {
  config: BibilabConfig;
  dependencies: Record<string, HealthDependency>;
  onBlur: (updated: BibilabConfig) => void;
};

export function TranscriptTab({ config, dependencies, onBlur }: TranscriptTabProps) {
  const { t } = useLanguage();
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

  async function refreshModels(signal?: AbortSignal) {
    try {
      const nextModels = await api.listWhisperModels({ signal });
      setModels(nextModels ?? []);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      setModels([]);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    void refreshModels(controller.signal);
    return () => controller.abort();
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
  }, [terminalModelJobCount]);

  function handleBlur() {
    onBlur({ ...config, transcription: localTranscription });
  }

  async function handleDownload(modelName: string) {
    setDownloading(modelName);
    try {
      const response = await api.downloadWhisperModel(modelName);
      if (!response) return;
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
    ? t("settings.cudaAvailable")
    : cudaDependency?.message ?? t("settings.cudaUnavailable");

  return (
    <div className="grid gap-4">
      <div className="flex flex-col gap-3">
        <SettingsField
          label={t("settings.modelSize")}
          hint={t("settings.modelSizeRequired")}
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
                {localTranscription.model_size} {t("settings.downloadRequired")}
              </option>
            ) : null}
            {installedModels.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </Select>
        </SettingsField>

        <SettingsField label={t("settings.device")} hint={deviceHint} htmlFor={deviceId}>
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
          label={t("settings.language")}
          hint={t("settings.languageDesc")}
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
          {t("settings.modelDownloads")}
        </p>
        <p className="text-sm leading-5 text-muted">{t("settings.modelDownloadsRequired")}</p>
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
                          t={t}
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
