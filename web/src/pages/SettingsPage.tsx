import { useEffect, useState } from "react";

import { ConfigForm } from "../components/settings/ConfigForm";
import { HealthPanel } from "../components/settings/HealthPanel";
import { WhisperModelsCard } from "../components/settings/WhisperModelsCard";
import { api } from "../lib/api";
import type { HealthResponse, LocusConfig, WhisperModel } from "../lib/types";

export function SettingsPage() {
  const [config, setConfig] = useState<LocusConfig | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<WhisperModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshingHealth, setRefreshingHealth] = useState(false);

  async function refreshHealth() {
    setRefreshingHealth(true);
    try {
      setHealth(await api.getHealth());
    } finally {
      setRefreshingHealth(false);
    }
  }

  async function refreshModels() {
    setModels(await api.listWhisperModels());
  }

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const [nextConfig, nextHealth, nextModels] = await Promise.all([
        api.getConfig(),
        api.getHealth(),
        api.listWhisperModels(),
      ]);
      if (!cancelled) {
        setConfig(nextConfig);
        setHealth(nextHealth);
        setModels(nextModels);
        setLoading(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSave(patch: Partial<LocusConfig>) {
    const nextConfig = await api.putConfig(patch);
    setConfig(nextConfig);
  }

  async function handleDownload(modelSize: string) {
    await api.downloadWhisperModel(modelSize);
    await refreshModels();
  }

  if (loading || !config || !health) {
    return (
      <section className="panel">
        <p>Loading settings...</p>
      </section>
    );
  }

  return (
    <div className="form-stack">
      <section>
        <h1 className="page-heading">Settings</h1>
        <p className="page-lede">Configure accounts, model choices, local dependencies, and downloads.</p>
      </section>
      <div className="settings-grid">
        <ConfigForm config={config} onSave={handleSave} />
        <HealthPanel health={health} onRefresh={refreshHealth} refreshing={refreshingHealth} />
        <WhisperModelsCard models={models} onDownload={handleDownload} />
      </div>
    </div>
  );
}
