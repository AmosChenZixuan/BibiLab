import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { LlmTab } from "@/components/settings/LlmTab";
import { OtherTab } from "@/components/settings/OtherTab";
import { TranscriptTab } from "@/components/settings/TranscriptTab";
import { api, notifyHealthChanged, toErrorMessage } from "@/lib/api";
import { deriveDependencyHealthTier, HEALTH_META } from "@/lib/health";
import type { HealthDependency, LocusConfig } from "@/lib/types";
import { Panel } from "@/components/ui";

type TabKey = "llm" | "transcript" | "other";

const TABS: Array<{ key: TabKey; label: string; dependencyKeys: string[] }> = [
  { key: "llm", label: "LLM", dependencyKeys: ["llm"] },
  { key: "transcript", label: "Transcript", dependencyKeys: ["whisper_model", "cuda"] },
  {
    key: "other",
    label: "Other",
    dependencyKeys: ["backend", "ffmpeg", "embedding_model"],
  },
];

function isTabKey(value: string | null): value is TabKey {
  return value === "llm" || value === "transcript" || value === "other";
}

function hasConfigChanged(current: LocusConfig, next: LocusConfig) {
  return JSON.stringify(current) !== JSON.stringify(next);
}

function shouldRefreshHealth(current: LocusConfig, next: LocusConfig) {
  return (
    current.ai.provider !== next.ai.provider ||
    current.ai.model !== next.ai.model ||
    current.ai.api_key !== next.ai.api_key ||
    current.ai.base_url !== next.ai.base_url ||
    current.transcription.model_size !== next.transcription.model_size ||
    current.transcription.device !== next.transcription.device
  );
}

export function SettingsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [config, setConfig] = useState<LocusConfig | null>(null);
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
    let cancelled = false;
    async function load() {
      try {
        const [nextConfig, nextHealth] = await Promise.all([
          api.getConfig(),
          api.getHealth(),
        ]);
        if (!cancelled) {
          setConfig(nextConfig);
          setDependencies(nextHealth.dependencies ?? {});
          setLoadError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(toErrorMessage(error));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSave(nextConfig: LocusConfig) {
    if (!config || !hasConfigChanged(config, nextConfig)) {
      return;
    }

    const savedConfig = await api.putConfig(nextConfig);
    setConfig(savedConfig);

    if (shouldRefreshHealth(config, nextConfig)) {
      const nextHealth = await api.getHealth();
      setDependencies(nextHealth.dependencies ?? {});
      notifyHealthChanged(nextHealth);
    }
  }

  if (loading) {
    return (
      <Panel variant="app">
        <p>Loading settings...</p>
      </Panel>
    );
  }

  if (loadError || !config) {
    return (
      <Panel variant="app">
        <h1 className="m-0 mb-2 font-semibold text-4xl leading-none md:text-5xl xl:text-6xl">Settings</h1>
        <p className="m-0 text-sm text-rose-900">{loadError ?? "Request failed"}</p>
      </Panel>
    );
  }

  return (
    <div className="grid gap-4">
      <section>
        <h1 className="m-0 mb-2 font-semibold text-4xl leading-none md:text-5xl xl:text-6xl">Settings</h1>
      </section>

      <section className="grid items-start gap-5 md:grid-cols-5">
        <div className="flex flex-col gap-1 md:col-span-1">
          {TABS.map((tab) => {
            const healthTier = deriveDependencyHealthTier(dependencies, tab.dependencyKeys);
            const isActive = activeTab === tab.key;
            const healthMeta = HEALTH_META[healthTier];

            return (
              <button
                key={tab.key}
                role="tab"
                aria-selected={isActive}
                className={`flex items-center gap-3 rounded-xl px-4 py-3 text-left transition ${
                  isActive
                    ? "bg-sky-50 font-semibold text-ink"
                    : "text-muted hover:bg-sky-50 hover:text-ink"
                }`}
                title={healthMeta.label}
                onClick={() => setActiveTab(tab.key)}
                type="button"
              >
                <span
                  className={`h-2.5 w-2.5 rounded-full ${healthMeta.className}`}
                  aria-hidden="true"
                />
                <span>{tab.label}</span>
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
