import { useCallback, useEffect, useState } from "react";
import { Download } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import type { ModelInfo, ModelKind } from "@/lib/types";
import { Spinner, Button } from "@/components/ui";
import { useJobActivity, type JobActivityItem } from "@/components/jobs/JobActivityProvider";
import { useRefreshOnTerminalModelJobs } from "@/components/jobs/useRefreshOnTerminalModelJobs";
import { formatBundleSize } from "@/lib/utils";

type ModelEntry = ModelInfo & { jobStatus?: "downloading" | "failed" | null };

const TRANSCRIPTION_KINDS: ReadonlyArray<ModelKind> = ["transcription", "vad", "diarization", "punctuation"];
const KIND_ORDER: Partial<Record<ModelKind, number>> = { transcription: 0, vad: 1, diarization: 2, punctuation: 3 };

function mergeJobStatus(entries: ModelInfo[], modelJobs: JobActivityItem[]): ModelEntry[] {
  return entries.map((m) => {
    const activeJob = modelJobs.find(
      (j) => "model_name" in j.job.meta && j.job.meta.model_name === m.id,
    );
    if (!activeJob || activeJob.isTerminal) return m;
    if (activeJob.job.status === "failed" || activeJob.job.status === "needs_auth") {
      return { ...m, jobStatus: "failed" };
    }
    return { ...m, jobStatus: "downloading" };
  });
}

function sortTranscription(entries: ModelEntry[]): ModelEntry[] {
  return [...entries].sort((a, b) => {
    const k = (KIND_ORDER[a.kind] ?? 9) - (KIND_ORDER[b.kind] ?? 9);
    if (k !== 0) return k;
    return (b.required_by_config ? 1 : 0) - (a.required_by_config ? 1 : 0);
  });
}

export function ModelsTab() {
  const { t } = useLanguage();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [syncing, setSyncing] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const { getJobs, trackJobs } = useJobActivity();

  const refreshModels = useCallback(async (signal?: AbortSignal) => {
    try {
      const data = await api.listModels({ signal });
      if (data) setModels(data);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshModels(controller.signal);
    return () => controller.abort();
  }, [refreshModels]);

  useRefreshOnTerminalModelJobs(refreshModels);
  const modelJobs = getJobs("model_download");

  const entries = mergeJobStatus(models, modelJobs);
  const transcription = sortTranscription(entries.filter((m) => TRANSCRIPTION_KINDS.includes(m.kind)));
  const embedding = entries.filter((m) => m.kind === "embedding");
  const reranker = entries.filter((m) => m.kind === "reranker");
  const missingRequired = entries.some((m) => m.required_by_config && m.status === "missing");

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
        await refreshModels();
      }
    } finally {
      setSyncing(false);
    }
  }

  if (entries.length === 0) {
    return <p className="text-sm text-muted">{t("settings.noModels")}</p>;
  }

  return (
    <div className="grid gap-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">{t("settings.models")}</h3>
        <Button disabled={syncing || !missingRequired} onClick={() => void handleSync()} size="sm" variant="primary">
          {syncing ? <Spinner label={t("settings.installing")} /> : t("settings.installAll")}
        </Button>
      </div>

      <ModelSection
        title={t("settings.kindTranscription")}
        hint={t("settings.kindTranscriptionHint")}
        entries={transcription}
        downloading={downloading}
        onDownload={handleDownload}
      />
      {embedding.length > 0 && (
        <ModelSection
          title={t("settings.kindEmbedding")}
          entries={embedding}
          downloading={downloading}
          onDownload={handleDownload}
        />
      )}
      {reranker.length > 0 && (
        <ModelSection
          title={t("settings.kindReranker")}
          entries={reranker}
          downloading={downloading}
          onDownload={handleDownload}
        />
      )}
    </div>
  );
}

type SectionProps = {
  title: string;
  hint?: string;
  entries: ModelEntry[];
  downloading: string | null;
  onDownload: (specId: string) => void;
};

function ModelSection({ title, hint, entries, downloading, onDownload }: SectionProps) {
  if (entries.length === 0) return null;
  return (
    <div className="grid gap-2">
      <p className="text-xs font-semibold uppercase tracking-widest text-muted">{title}</p>
      {hint && <p className="text-sm text-muted">{hint}</p>}
      <div className="overflow-hidden rounded-2xl border border-border bg-white/64">
        {entries.map((m, i) => (
          <ModelRow
            key={m.id}
            model={m}
            downloading={downloading}
            onDownload={onDownload}
            isFirst={i === 0}
          />
        ))}
      </div>
    </div>
  );
}

type RowProps = {
  model: ModelEntry;
  downloading: string | null;
  onDownload: (specId: string) => void;
  isFirst: boolean;
};

function ModelRow({ model, downloading, onDownload, isFirst }: RowProps) {
  const { t } = useLanguage();
  const isPresent = model.status === "present";
  const isEngine = model.kind === "transcription";
  const engineBadge = isEngine
    ? model.required_by_config
      ? t("settings.currentEngine")
      : t("settings.alternateEngine")
    : null;

  return (
    <div className={`flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 ${isFirst ? "" : "border-t border-border"}`}>
      <div className="flex min-w-48 flex-1 items-baseline gap-2 flex-wrap">
        <span className="text-sm font-semibold text-ink">{model.display_name}</span>
        {engineBadge && <span className="text-xs text-muted">· {engineBadge}</span>}
      </div>
      <span className="font-mono text-xs text-muted whitespace-nowrap">{formatBundleSize(model.size_mb)}</span>
      <div className="flex min-w-40 flex-1 items-center justify-end">
        {model.jobStatus === "downloading" ? (
          <Spinner label={t("common.downloading", { name: model.display_name })} />
        ) : model.jobStatus === "failed" ? (
          <span className="text-xs text-pink">{t("common.failed")}</span>
        ) : isPresent ? (
          <p className="text-right font-mono text-xs text-muted break-all" title={model.path ?? ""}>
            {model.path ?? "—"}
          </p>
        ) : (
          <button
            type="button"
            className="inline-flex items-center justify-center text-sky disabled:opacity-50"
            aria-label={t("common.download")}
            disabled={downloading === model.id}
            onClick={() => onDownload(model.id)}
          >
            <Download size={18} />
          </button>
        )}
      </div>
    </div>
  );
}
