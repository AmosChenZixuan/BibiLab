import { useEffect, useId, useState } from "react";

import { useLanguage, type Lang } from "@/app/LanguageContext";
import type { HealthDependency, BibilabConfig } from "@/lib/types";
import { Input, Select, SettingsField } from "@/components/ui";

type OtherTabProps = {
  config: BibilabConfig;
  dependencies: Record<string, HealthDependency>;
  onBlur: (updated: BibilabConfig) => void;
};

export function OtherTab({ config, dependencies, onBlur }: OtherTabProps) {
  const { lang, setLang, t } = useLanguage();
  const [local, setLocal] = useState({
    workerConcurrency: config.backend.worker_concurrency,
  });
  const interfaceLanguageId = useId();
  const workerConcurrencyId = useId();

  useEffect(() => {
    setLocal({
      workerConcurrency: config.backend.worker_concurrency,
    });
  }, [config.backend.worker_concurrency]);

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
  const ffmpegDependency = dependencies.ffmpeg;
  const ffmpegOk = ffmpegDependency?.status === "ok";
  const backendOk = dependencies.backend?.status === "ok";

  const valueClass = "text-right font-mono text-sm text-muted break-all";

  return (
    <div className="grid gap-4">
      <SettingsField
        label={t("settings.interfaceLanguage")}
        hint={t("settings.interfaceLanguageDesc")}
        htmlFor={interfaceLanguageId}
      >
        <Select
          aria-label={t("settings.interfaceLanguage")}
          id={interfaceLanguageId}
          onChange={(event) => setLang(event.target.value as Lang)}
          value={lang}
        >
          <option value="en">{t("settings.interfaceLanguageEn")}</option>
          <option value="zh">{t("settings.interfaceLanguageZh")}</option>
        </Select>
      </SettingsField>

      <SettingsField label={t("settings.backendApi")} hint={t("settings.backendRequired")}>
        <p className={valueClass}>{backendOk ? backendUrl : t("settings.backendOffline")}</p>
      </SettingsField>

      <SettingsField
        label={t("settings.workerConcurrency")}
        hint={t("settings.workerConcurrencyDesc")}
        htmlFor={workerConcurrencyId}
      >
        <Input
          aria-label={t("settings.workerConcurrency")}
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
      </SettingsField>

      <SettingsField label={t("settings.ffmpeg")} hint={t("settings.ffmpegRequired")}>
        {ffmpegOk ? (
          <p className={valueClass}>{ffmpegDependency?.message ?? "—"}</p>
        ) : (
          <div className="flex flex-wrap items-center justify-end gap-3 text-sm">
            <span className="text-pink">{t("settings.ffmpegOffline")}</span>
            <a
              className="text-blue underline"
              href="https://ffmpeg.org/download.html"
              rel="noreferrer"
              target="_blank"
            >
              {t("settings.downloadFfmpeg")}
            </a>
          </div>
        )}
      </SettingsField>
    </div>
  );
}
