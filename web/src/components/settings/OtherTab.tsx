import { useEffect, useId, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import type { HealthDependency, BibilabConfig } from "@/lib/types";
import { Input, StatusChip } from "@/components/ui";

type OtherTabProps = {
  config: BibilabConfig;
  dependencies: Record<string, HealthDependency>;
  onBlur: (updated: BibilabConfig) => void;
};

export function OtherTab({ config, dependencies, onBlur }: OtherTabProps) {
  const { t } = useLanguage();
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

  const valueClass = "ml-auto text-right font-mono text-sm text-muted";

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 bg-white/36 px-4 py-3">
        <div className="grid min-w-48 flex-1 basis-60 gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-semibold">{t("settings.embeddingModel")}</p>
            <StatusChip status={embeddingDependency?.status === "ok" ? "ok" : "error"}>
              {t(embeddingDependency?.status === "ok" ? "settings.ready" : "settings.missing")}
            </StatusChip>
          </div>
          <p className="text-sm leading-5 text-muted">{t("settings.embeddingModelMissing")}</p>
        </div>
        <div className="flex min-w-56 flex-1 items-center justify-end gap-3 self-center">
          {embeddingDependency?.status === "ok" ? (
            <p className={valueClass}>
              {embeddingPath}
            </p>
          ) : (
            <span className="max-w-3xl text-right text-sm leading-6 text-blue">
              {embeddingDependency?.message}
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-x-5 gap-y-2 bg-white/36 px-4 py-3 md:grid-cols-5">
        <div className="flex flex-wrap items-center gap-2 md:col-span-3">
          <p className="text-sm font-semibold">{t("settings.backendApi")}</p>
          <StatusChip status={backendDependency?.status === "ok" ? "ok" : "error"}>
            {t(backendDependency?.status === "ok" ? "settings.connected" : "settings.offline")}
          </StatusChip>
        </div>
        <div className="flex items-center justify-end md:col-span-2">
          <p className={valueClass}>{backendUrl}</p>
        </div>

        <p className="text-sm leading-5 text-muted md:col-span-3">{t("settings.backendRequired")}</p>
        <div />

        <div className="border-l border-blue/18 pl-4 md:col-span-3">
          <div className="grid gap-2">
            <label className="text-sm font-semibold" htmlFor={workerConcurrencyId}>{t("settings.workerConcurrency")}</label>
            <p className="text-sm leading-5 text-muted">{t("settings.workerConcurrencyDesc")}</p>
          </div>
        </div>
        <div className="flex items-center justify-end md:col-span-2">
          <Input
            aria-label="Worker Concurrency"
            className="min-w-56 flex-none md:w-80"
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
            inputSize="sm"
            type="number"
            value={local.workerConcurrency}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 bg-white/36 px-4 py-3">
        <div className="grid min-w-48 flex-1 basis-60 gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-semibold">{t("settings.ffmpeg")}</p>
            <StatusChip
              status={ffmpegDependency?.status === "ok" ? "ok" : "error"}
              title={ffmpegDependency?.status === "ok" ? t("settings.ffmpegInstalled") : t("settings.ffmpegNotFound")}
            >
              {t(ffmpegDependency?.status === "ok" ? "settings.installed" : "settings.missing")}
            </StatusChip>
          </div>
          <p className="text-sm leading-5 text-muted">{t("settings.ffmpegRequired")}</p>
        </div>
        <div className="flex min-w-56 flex-1 items-center justify-end gap-3 self-center">
          {ffmpegDependency?.status === "ok" ? (
            <p className={valueClass}>
              {ffmpegPath}
            </p>
          ) : (
            <a
              className="text-blue underline"
              href="https://ffmpeg.org/download.html"
              rel="noreferrer"
              target="_blank"
            >
              {t("settings.downloadFfmpeg")}
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
