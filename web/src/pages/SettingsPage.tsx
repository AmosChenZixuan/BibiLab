import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useLanguage } from "@/app/LanguageContext";
import { LlmTab } from "@/components/settings/LlmTab";
import { OtherTab } from "@/components/settings/OtherTab";
import { TranscriptTab } from "@/components/settings/TranscriptTab";
import { api, notifyHealthChanged, toErrorMessageWithT } from "@/lib/api";
import { deriveDependencyHealthTier, HEALTH_META } from "@/lib/health";
import type { HealthDependency, BibilabConfig } from "@/lib/types";
import { Panel } from "@/components/ui";

type TabKey = "llm" | "transcript" | "other";

function isTabKey(value: string | null): value is TabKey {
  return value === "llm" || value === "transcript" || value === "other";
}

function hasConfigChanged(current: BibilabConfig, next: BibilabConfig) {
  return JSON.stringify(current) !== JSON.stringify(next);
}

function shouldRefreshHealth(current: BibilabConfig, next: BibilabConfig) {
  return (
    current.ai.protocol !== next.ai.protocol ||
    current.ai.model !== next.ai.model ||
    current.ai.api_key !== next.ai.api_key ||
    current.ai.base_url !== next.ai.base_url ||
    current.transcription.model_size !== next.transcription.model_size ||
    current.transcription.device !== next.transcription.device
  );
}

const TABS: ReadonlyArray<{ key: TabKey; labelKey: string; dependencyKeys: readonly string[] }> = [
  { key: "llm", labelKey: "settings.llm", dependencyKeys: ["llm"] as const },
  { key: "transcript", labelKey: "settings.transcript", dependencyKeys: ["whisper_model"] as const },
  { key: "other", labelKey: "settings.other", dependencyKeys: ["backend", "ffmpeg", "embedding_model"] as const },
];

export function SettingsPage() {
  const { t } = useLanguage();
  const [searchParams, setSearchParams] = useSearchParams();
  const [config, setConfig] = useState<BibilabConfig | null>(null);
  const [dependencies, setDependencies] = useState<Record<string, HealthDependency>>({});
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const activeTab = useMemo<TabKey>(() => {
    const tab = searchParams.get("tab");
    return isTabKey(tab) ? tab : "llm";
  }, [searchParams]);

  function setActiveTab(tab: TabKey) {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set("tab", tab);
    setSearchParams(nextParams, { replace: true });
  }

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const [nextConfig, nextHealth] = await Promise.all([
          api.getConfig({ signal: controller.signal }),
          api.getHealth({ signal: controller.signal }),
        ]);
        setConfig(nextConfig ?? null);
        if (nextHealth) {
          setDependencies(nextHealth.dependencies ?? {});
        }
        setLoadError(null);
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") return;
        setLoadError(toErrorMessageWithT(error, t));
      } finally {
        setLoading(false);
      }
    }

    void load();
    return () => controller.abort();
  }, [t]);

  async function handleSave(nextConfig: BibilabConfig) {
    if (!config || !hasConfigChanged(config, nextConfig)) {
      return;
    }

    const savedConfig = await api.putConfig(nextConfig);
    if (!savedConfig) return;
    setConfig(savedConfig);

    if (shouldRefreshHealth(config, nextConfig)) {
      const nextHealth = await api.getHealth();
      if (nextHealth) {
        setDependencies(nextHealth.dependencies ?? {});
        notifyHealthChanged(nextHealth, savedConfig.transcription.device);
      }
    }
  }

  if (loading) {
    return (
      <Panel variant="app">
        <p>{t("settings.loading")}</p>
      </Panel>
    );
  }

  if (loadError || !config) {
    return (
      <Panel variant="app">
        <h1 className="m-0 mb-2 font-semibold text-4xl leading-none md:text-5xl xl:text-6xl">{t("settings.title")}</h1>
        <p className="m-0 text-sm text-pink">{loadError ?? t("errors.requestFailed")}</p>
      </Panel>
    );
  }

  return (
    <div className="grid gap-4">
      <section>
        <h1 className="m-0 mb-2 font-semibold text-4xl leading-none md:text-5xl xl:text-6xl">{t("settings.title")}</h1>
      </section>

      <section className="grid items-start gap-5 md:grid-cols-5">
        <div className="flex flex-col gap-1 md:col-span-1">
          {TABS.map((tab) => {
            let healthTier = deriveDependencyHealthTier(dependencies, tab.dependencyKeys);
            if (
              tab.key === "transcript" &&
              healthTier === "healthy" &&
              dependencies.cuda?.status === "ok" &&
              config?.transcription.device !== "cuda"
            ) {
              healthTier = "throttled";
            }
            const isActive = activeTab === tab.key;

            return (
              <button
                key={tab.key}
                role="tab"
                aria-selected={isActive}
                className={`flex items-center gap-3 rounded-xl px-4 py-3 text-left transition ${
                  isActive
                    ? "bg-sky/10 font-semibold text-ink"
                    : "text-muted hover:bg-sky/10 hover:text-ink"
                }`}
                title={t("health." + healthTier)}
                onClick={() => setActiveTab(tab.key)}
                type="button"
              >
                <span
                  className={`h-2.5 w-2.5 rounded-full ${HEALTH_META[healthTier].className}`}
                  aria-hidden="true"
                />
                <span>{t(tab.labelKey)}</span>
              </button>
            );
          })}
        </div>

        <div className="min-w-0 md:col-span-4">
          {activeTab === "llm" ? <LlmTab config={config} onBlur={handleSave} /> : null}
          {activeTab === "transcript" ? (
            <TranscriptTab config={config} dependencies={dependencies} onBlur={handleSave} />
          ) : null}
          {activeTab === "other" ? (
            <OtherTab config={config} dependencies={dependencies} onBlur={handleSave} />
          ) : null}
        </div>
      </section>
    </div>
  );
}
