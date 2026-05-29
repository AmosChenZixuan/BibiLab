import { useEffect, useId, useState } from "react";
import { Download } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import type { ModelInfo } from "@/lib/types";
import { Spinner, StatusChip, Button } from "@/components/ui";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { formatBundleSize } from "./TranscriptTab";

type ModelEntry = ModelInfo & { jobStatus?: "downloading" | "failed" | null };

export function ModelsTab() {
  const { t } = useLanguage();
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [syncing, setSyncing] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const { getJobs, trackJobs, dismissJob } = useJobActivity();
  const tableId = useId();

  const modelJobs = getJobs("model_download");

  useEffect(() => {
    let cancelled = false;
    api.listModels().then((data) => {
      if (!cancelled && data) setModels(data);
    });
    return () => { cancelled = true; };
  }, []);

  // Refresh model list when model_download jobs reach terminal state
  useEffect(() => {
    const terminalJobs = modelJobs.filter((j) => j.isTerminal);
    if (terminalJobs.length === 0) return undefined;

    api.listModels().then((data) => {
      if (data) setModels(data);
    });
    for (const j of terminalJobs) dismissJob(j.job.id);
    return undefined;
  }, [modelJobs]);

  function mergeJobStatus(entries: ModelEntry[]): ModelEntry[] {
    return entries.map((m) => {
      const activeJob = modelJobs.find(
        (j) => "model_name" in j.job.meta && j.job.meta.model_name === m.id
      );
      if (!activeJob || activeJob.isTerminal) return m;
      if (activeJob.job.status === "failed" || activeJob.job.status === "needs_auth") {
        return { ...m, jobStatus: "failed" };
      }
      return { ...m, jobStatus: "downloading" };
    });
  }

  async function handleDownload(specId: string) {
    setDownloading(specId);
    try {
      const resp = await api.downloadModel(specId);
      if (resp) {
        trackJobs([{ id: resp.job_id, producer: "model_download", label: specId, contextKey: specId }]);
      }
    } finally {
      setDownloading(null);
    }
  }

  async function handleSync() {
    setSyncing(true);
    try {
      const resp = await api.syncModels();
      if (resp) {
        for (let i = 0; i < resp.job_ids.length; i++) {
          trackJobs([{ id: resp.job_ids[i], producer: "model_download", label: resp.synced[i] ?? "" }]);
        }
        const data = await api.listModels();
        if (data) setModels(data);
      }
    } finally {
      setSyncing(false);
    }
  }

  const entries = mergeJobStatus(models);
  const requiredEntries = entries.filter((m) => m.required_by_config);
  const optionalEntries = entries.filter((m) => !m.required_by_config);

  return (
    <div className="grid gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">{t("settings.requiredModels", { count: requiredEntries.length })}</h3>
        <Button disabled={syncing} onClick={() => void handleSync()} size="sm" variant="primary">
          {syncing ? <Spinner label={t("settings.syncingModels")} /> : t("settings.syncRequired")}
        </Button>
      </div>

      <table className="w-full text-sm" aria-label={t("settings.modelsTable")} id={tableId}>
        <thead>
          <tr className="text-left text-xs uppercase text-muted">
            <th className="pb-2 font-medium">{t("settings.model")}</th>
            <th className="pb-2 font-medium">{t("settings.kind")}</th>
            <th className="pb-2 font-medium">{t("settings.size")}</th>
            <th className="pb-2 font-medium">{t("settings.status")}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {entries.length === 0 && (
            <tr><td className="py-6 text-center text-muted" colSpan={5}>{t("settings.noModels")}</td></tr>
          )}
          {requiredEntries.map((m) => (
            <ModelRow key={m.id} model={m} downloading={downloading} onDownload={(id) => void handleDownload(id)} />
          ))}
          {optionalEntries.length > 0 && requiredEntries.length > 0 && (
            <tr className="border-none">
              <td className="pb-1 pt-4 text-xs font-semibold text-muted" colSpan={5}>
                {t("settings.optionalModels")}
              </td>
            </tr>
          )}
          {optionalEntries.map((m) => (
            <ModelRow key={m.id} model={m} downloading={downloading} onDownload={(id) => void handleDownload(id)} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelRow({
  model,
  downloading,
  onDownload,
}: {
  model: ModelEntry;
  downloading: string | null;
  onDownload: (id: string) => void;
}) {
  const { t } = useLanguage();
  const isPresent = model.status === "present";

  return (
    <tr key={model.id}>
      <td className="py-2 pr-4 font-medium">{model.display_name}</td>
      <td className="py-2 pr-4 text-xs text-muted">{model.kind}</td>
      <td className="py-2 pr-4 font-mono text-xs text-muted">{formatBundleSize(model.size_mb)}</td>
      <td className="py-2">
        {model.jobStatus === "downloading" ? (
          <Spinner label={t("common.downloading", { name: model.display_name })} />
        ) : model.jobStatus === "failed" ? (
          <span className="text-xs text-pink">{t("common.failed")}</span>
        ) : isPresent ? (
          <StatusChip status="ok">{t("settings.ready")}</StatusChip>
        ) : (
          <Button
            disabled={downloading === model.id}
            onClick={() => onDownload(model.id)}
            size="sm"
            variant="ghost"
          >
            <Download className="mr-1 size-3.5" />
            {t("settings.download")}
          </Button>
        )}
      </td>
    </tr>
  );
}
