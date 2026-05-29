import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import type { BibilabConfig, HealthDependency, ModelInfo } from "@/lib/types";

import { Select, SettingsField } from "@/components/ui";

type TranscriptTabProps = {
  config: BibilabConfig;
  dependencies: Record<string, HealthDependency>;
  onBlur: (updated: BibilabConfig) => void;
};

export function TranscriptTab({ config, dependencies, onBlur }: TranscriptTabProps) {
  const { t } = useLanguage();
  const { dismissJob, getJobs } = useJobActivity();
  const [localTranscription, setLocalTranscription] = useState(config.transcription);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const modelId = useId();
  const deviceId = useId();
  const languageId = useId();

  useEffect(() => {
    setLocalTranscription(config.transcription);
  }, [config.transcription]);

  const refreshModels = useCallback(async (signal?: AbortSignal) => {
    try {
      const nextModels = await api.listModels({ signal });
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
  }, [refreshModels]);

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

  const installedTranscriptionModels = useMemo(
    () => models.filter((m) => m.kind === "transcription" && m.status === "present"),
    [models],
  );

  const cudaSupported = dependencies.cuda?.status === "ok";
  const noInstalled = installedTranscriptionModels.length === 0;

  return (
    <div className="grid gap-4">
      <SettingsField
        label={t("settings.transcriptionModel")}
        hint={noInstalled ? undefined : t("settings.transcriptionModelRequired")}
        htmlFor={modelId}
      >
        <div className="grid gap-2">
          <Select
            aria-label="Model"
            id={modelId}
            disabled={noInstalled}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalTranscription((current) => ({ ...current, model: event.target.value }))
            }
            value={localTranscription.model}
          >
            {noInstalled ? (
              <option disabled value={localTranscription.model}>
                {t("settings.noInstalledModel")}
              </option>
            ) : (
              installedTranscriptionModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.display_name}
                </option>
              ))
            )}
          </Select>
          {noInstalled && (
            <Link to="/settings?tab=models" className="text-sm text-blue underline">
              {t("settings.manageModels")}
            </Link>
          )}
        </div>
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

      <SettingsField
        label={t("settings.transcriptionLanguage")}
        hint={t("settings.transcriptionLanguageDesc")}
        htmlFor={languageId}
      >
        <Select
          aria-label="Transcription Language"
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
  );
}
