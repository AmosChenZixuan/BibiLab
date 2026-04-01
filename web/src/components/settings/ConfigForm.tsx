import { useEffect, useState } from "react";

import type { LocusConfig } from "../../lib/types";
import {
  appPanelClass,
  checkboxRowClass,
  fieldClass,
  fieldLabelClass,
  inputClass,
  mutedTextClass,
  primaryButtonClass,
  sectionTitleClass,
  statusSuccessClass,
  textareaClass,
} from "../../lib/ui";

const MASK = "***";

type Props = {
  config: LocusConfig;
  onSave: (patch: Partial<LocusConfig>) => Promise<void>;
};

function buildPatch(initial: LocusConfig, current: LocusConfig): Partial<LocusConfig> {
  const patch: Partial<LocusConfig> = {};

  if (
    current.accounts.bilibili.cookie !== initial.accounts.bilibili.cookie &&
    current.accounts.bilibili.cookie !== MASK
  ) {
    patch.accounts = {
      bilibili: {
        cookie: current.accounts.bilibili.cookie,
        last_verified: current.accounts.bilibili.last_verified,
      },
    };
  }

  const aiPatch: Partial<LocusConfig["ai"]> = {};
  if (current.ai.provider !== initial.ai.provider) aiPatch.provider = current.ai.provider;
  if (current.ai.model !== initial.ai.model) aiPatch.model = current.ai.model;
  if (current.ai.base_url !== initial.ai.base_url) aiPatch.base_url = current.ai.base_url;
  if (current.ai.api_key !== initial.ai.api_key && current.ai.api_key !== MASK) {
    aiPatch.api_key = current.ai.api_key;
  }
  if (Object.keys(aiPatch).length > 0) patch.ai = aiPatch as LocusConfig["ai"];

  const transcriptionPatch: Partial<LocusConfig["transcription"]> = {};
  if (current.transcription.model_size !== initial.transcription.model_size) {
    transcriptionPatch.model_size = current.transcription.model_size;
  }
  if (current.transcription.device !== initial.transcription.device) {
    transcriptionPatch.device = current.transcription.device;
  }
  if (current.transcription.language !== initial.transcription.language) {
    transcriptionPatch.language = current.transcription.language;
  }
  if (Object.keys(transcriptionPatch).length > 0) {
    patch.transcription = transcriptionPatch as LocusConfig["transcription"];
  }

  const visionPatch: Partial<LocusConfig["vision"]> = {};
  if (current.vision.enabled !== initial.vision.enabled) visionPatch.enabled = current.vision.enabled;
  if (current.vision.frame_sample_rate !== initial.vision.frame_sample_rate) {
    visionPatch.frame_sample_rate = current.vision.frame_sample_rate;
  }
  if (current.vision.model !== initial.vision.model) visionPatch.model = current.vision.model;
  if (Object.keys(visionPatch).length > 0) patch.vision = visionPatch as LocusConfig["vision"];

  if (current.backend.worker_concurrency !== initial.backend.worker_concurrency) {
    patch.backend = {
      worker_concurrency: current.backend.worker_concurrency,
      port: initial.backend.port,
    };
  }

  return patch;
}

export function ConfigForm({ config, onSave }: Props) {
  const [draft, setDraft] = useState<LocusConfig>(config);
  const [status, setStatus] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDraft(config);
    setStatus(null);
  }, [config]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setStatus(null);
    try {
      await onSave(buildPatch(config, draft));
      setStatus("Settings saved");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className={`${appPanelClass} col-span-2 grid gap-4 max-[820px]:col-span-1`}>
      <div className="flex flex-wrap items-center gap-3">
        <div>
          <h2 className={sectionTitleClass}>Configuration</h2>
          <p className={mutedTextClass}>Keep secrets local and patch only what changed.</p>
        </div>
      </div>
      <form className="grid gap-4" onSubmit={handleSubmit}>
        <div className="grid grid-cols-2 gap-4 max-[820px]:grid-cols-1">
          <label className={fieldClass}>
            <span className={fieldLabelClass}>Bilibili cookie</span>
            <textarea
              className={textareaClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  accounts: {
                    bilibili: {
                      ...current.accounts.bilibili,
                      cookie: event.target.value,
                    },
                  },
                }))
              }
              rows={3}
              value={draft.accounts.bilibili.cookie}
            />
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>AI provider</span>
            <select
              className={inputClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  ai: { ...current.ai, provider: event.target.value },
                }))
              }
              value={draft.ai.provider}
            >
              <option value="openai">openai</option>
              <option value="anthropic">anthropic</option>
              <option value="ollama">ollama</option>
              <option value="custom">custom</option>
            </select>
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>AI model</span>
            <input
              aria-label="AI model"
              className={inputClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  ai: { ...current.ai, model: event.target.value },
                }))
              }
              value={draft.ai.model}
            />
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>AI API key</span>
            <input
              className={inputClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  ai: { ...current.ai, api_key: event.target.value },
                }))
              }
              value={draft.ai.api_key}
            />
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>AI base URL</span>
            <input
              className={inputClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  ai: { ...current.ai, base_url: event.target.value || null },
                }))
              }
              value={draft.ai.base_url ?? ""}
            />
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>Whisper model</span>
            <input
              className={inputClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  transcription: {
                    ...current.transcription,
                    model_size: event.target.value,
                  },
                }))
              }
              value={draft.transcription.model_size}
            />
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>Transcription device</span>
            <select
              className={inputClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  transcription: {
                    ...current.transcription,
                    device: event.target.value,
                  },
                }))
              }
              value={draft.transcription.device}
            >
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>Language</span>
            <select
              className={inputClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  transcription: {
                    ...current.transcription,
                    language: event.target.value,
                  },
                }))
              }
              value={draft.transcription.language}
            >
              <option value="auto">auto</option>
              <option value="zh">zh</option>
              <option value="en">en</option>
            </select>
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>Worker concurrency</span>
            <input
              className={inputClass}
              min={1}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  backend: {
                    ...current.backend,
                    worker_concurrency: Number(event.target.value),
                  },
                }))
              }
              type="number"
              value={draft.backend.worker_concurrency}
            />
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>Frame sample rate</span>
            <input
              className={inputClass}
              min={1}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  vision: {
                    ...current.vision,
                    frame_sample_rate: Number(event.target.value),
                  },
                }))
              }
              type="number"
              value={draft.vision.frame_sample_rate}
            />
          </label>
          <label className={fieldClass}>
            <span className={fieldLabelClass}>Vision model</span>
            <input
              className={inputClass}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  vision: {
                    ...current.vision,
                    model: event.target.value || null,
                  },
                }))
              }
              value={draft.vision.model ?? ""}
            />
          </label>
          <label className={checkboxRowClass}>
            <input
              checked={draft.vision.enabled}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  vision: {
                    ...current.vision,
                    enabled: event.target.checked,
                  },
                }))
              }
              type="checkbox"
            />
            <span>Vision enabled</span>
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button className={primaryButtonClass} disabled={saving} type="submit">
            {saving ? "Saving..." : "Save settings"}
          </button>
          {status ? <p className={statusSuccessClass}>{status}</p> : null}
        </div>
      </form>
    </section>
  );
}
