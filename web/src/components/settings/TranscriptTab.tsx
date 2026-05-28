import { Fragment, useCallback, useEffect, useId, useMemo, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import type { JobActivityItem } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import type { AsrModel, AsrModelKind, BibilabConfig, HealthDependency } from "@/lib/types";
import { Download } from "lucide-react";

import { Select, SettingsField, Spinner } from "@/components/ui";

const KIND_ORDER: ReadonlyArray<AsrModelKind> = ["transcription", "diarization"];

function formatBundleSize(sizeMb: number | null): string {
  if (sizeMb == null) return "—";
  if (sizeMb >= 1000) return `${(sizeMb / 1000).toFixed(1)} GB`;
  return `${sizeMb} MB`;
}

type ModelDownloadCellProps = {
  model: AsrModel;
  modelJob: JobActivityItem | null;
  downloading: string | null;
  onDownload: (modelName: string) => Promise<void>;
  t: (key: string, params?: Record<string, string | number>) => string;
};

function ModelDownloadCell({ model, modelJob, downloading, onDownload, t }: ModelDownloadCellProps) {
  if (modelJob && !modelJob.isTerminal) {
    return <Spinner label={t("common.downloading", { name: model.name })} />;
  }
  if (modelJob?.job.status === "failed" || modelJob?.job.status === "needs_auth") {
    return <span className="text-xs text-pink">{modelJob.job.error ?? t("common.failed")}</span>;
  }
  return (
    <button
      type="button"
      className="flex items-center justify-center text-sky"
      aria-label="Download"
      disabled={downloading === model.name}
      onClick={() => void onDownload(model.name)}
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
  const [models, setModels] = useState<AsrModel[]>([]);
  const [downloading, setDownloading] = useState<string | null>(null);
  const modelId = useId();
  const deviceId = useId();
  const languageId = useId();

  useEffect(() => {
    setLocalTranscription(config.transcription);
  }, [config.transcription]);

  const refreshModels = useCallback(async (signal?: AbortSignal) => {
    try {
      const nextModels = await api.listAsrModels({ signal });
      setModels(nextModels ?? []);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      setModels([]);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshModels(controller.signal);
    return () => controller.abort();
  }, []);

  const modelJobs = getJobs("model_download");
  const terminalModelJobCount = modelJobs.filter((j) => j.isTerminal).length;

  useEffect(() => {
    if (terminalModelJobCount === 0) return;
    const terminalIds = modelJobs.filter((j) => j.isTerminal).map((j) => j.job.id);
    async function refreshAndDismiss() {
      await refreshModels();
      for (const id of terminalIds) dismissJob(id);
    }
    void refreshAndDismiss();
  }, [modelJobs, terminalModelJobCount, dismissJob, refreshModels]);

  function handleBlur() {
    onBlur({ ...config, transcription: localTranscription });
  }

  async function handleDownload(modelName: string) {
    setDownloading(modelName);
    try {
      const response = await api.downloadAsrModel(modelName);
      if (!response) return;
      trackJobs([
        {
          id: response.job_id,
          producer: "model_download",
          label: modelName,
          contextKey: modelName,
        },
      ]);
    } finally {
      setDownloading(null);
    }
  }

  const installedTranscriptionModels = useMemo(
    () => models.filter((m) => m.kind === "transcription" && m.installed),
    [models],
  );
  const hasSelectedInstalled = useMemo(
    () => installedTranscriptionModels.some((m) => m.name === localTranscription.model),
    [installedTranscriptionModels, localTranscription.model],
  );

  const cudaSupported = dependencies.cuda?.status === "ok";

  const modelsByKind = useMemo(() => {
    const grouped: Record<string, AsrModel[]> = {};
    for (const m of models) {
      (grouped[m.kind] ??= []).push(m);
    }
    return KIND_ORDER.filter((k) => grouped[k]).map((k) => ({ kind: k, models: grouped[k] }));
  }, [models]);

  return (
    <div className="grid gap-4">
      <div className="flex flex-col gap-3">
        <SettingsField label={t("settings.model")} htmlFor={modelId}>
          <Select
            aria-label="Model"
            id={modelId}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalTranscription((current) => ({ ...current, model: event.target.value }))
            }
            value={localTranscription.model}
          >
            {!hasSelectedInstalled ? (
              <option disabled value={localTranscription.model}>
                {localTranscription.model} {t("settings.downloadRequired")}
              </option>
            ) : null}
            {installedTranscriptionModels.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </Select>
        </SettingsField>

        <SettingsField
          label={t("settings.device")}
          hint={cudaSupported ? t("settings.cudaAvailable") : t("settings.cudaUnavailable")}
          htmlFor={deviceId}
        >
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
            <option value="cuda" disabled={!cudaSupported}>CUDA</option>
          </Select>
        </SettingsField>

        <SettingsField label={t("settings.language")} hint={t("settings.languageDesc")} htmlFor={languageId}>
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
              {modelsByKind.map(({ kind, models: kindModels }) => (
                <Fragment key={kind}>
                  <tr className="border-t border-border">
                    <td colSpan={3} className="px-4 py-2 text-xs font-semibold text-muted uppercase">
                      {kind}
                    </td>
                  </tr>
                  {kindModels.map((model) => (
                    <tr key={model.name} className="border-t border-border">
                      <td className="px-4 py-3 pl-6 font-semibold text-ink">{model.name}</td>
                      <td className="px-4 py-3 font-mono text-sm text-muted whitespace-nowrap">
                        {formatBundleSize(model.size_mb)}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center justify-end">
                          {model.installed ? (
                            <p className="text-right font-mono text-sm text-muted">{model.path ?? "—"}</p>
                          ) : model.kind === "diarization" ? (
                            <span className="text-xs text-muted">{t("settings.autoInstalls")}</span>
                          ) : (
                            <ModelDownloadCell
                              model={model}
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
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
