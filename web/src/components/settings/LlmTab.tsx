import { useEffect, useId, useState } from "react";

import type { LocusConfig } from "../../lib/types";
import {
  fieldHintClass,
  fieldLabelClass,
  settingsControlClass,
  settingsFieldClass,
  settingsFieldMetaClass,
  settingsInputClass,
  settingsSelectClass,
} from "../../lib/ui";

type LlmTabProps = {
  config: LocusConfig;
  onBlur: (updated: LocusConfig) => void;
};

const BASE_URL_META: Record<string, { hint: string; placeholder: string }> = {
  openai: {
    hint: "Required. Use your OpenAI-compatible models endpoint, for example the official OpenAI v1 base URL.",
    placeholder: "https://api.openai.com/v1",
  },
  anthropic: {
    hint: "Required. Use the Anthropic API base URL that serves the models endpoint.",
    placeholder: "https://api.anthropic.com/v1",
  },
  ollama: {
    hint: "Required. Use your Ollama server root so health checks can reach /api/tags.",
    placeholder: "http://localhost:11434",
  },
  custom: {
    hint: "Required. Use your OpenAI-compatible provider base URL so health checks can reach /models.",
    placeholder: "http://localhost:8000/v1",
  },
};

export function LlmTab({ config, onBlur }: LlmTabProps) {
  const [localAi, setLocalAi] = useState(config.ai);
  const providerId = useId();
  const modelId = useId();
  const apiKeyId = useId();
  const baseUrlId = useId();

  useEffect(() => {
    setLocalAi(config.ai);
  }, [config]);

  const baseUrlMeta = BASE_URL_META[localAi.provider] ?? BASE_URL_META.custom;

  function handleBlur() {
    onBlur({ ...config, ai: localAi });
  }

  return (
    <div className="grid gap-3">
      <div className={settingsFieldClass}>
        <div className={settingsFieldMetaClass}>
          <label className={fieldLabelClass} htmlFor={providerId}>Provider</label>
          <p className={fieldHintClass}>Required. Without a provider, LLM requests cannot start.</p>
        </div>
        <select
          aria-label="Provider"
          className={settingsSelectClass}
          id={providerId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({
              ...current,
              provider: event.target.value,
            }))
          }
          value={localAi.provider}
        >
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
          <option value="ollama">Ollama</option>
          <option value="custom">Custom</option>
        </select>
      </div>

      <div className={settingsFieldClass}>
        <div className={settingsFieldMetaClass}>
          <label className={fieldLabelClass} htmlFor={modelId}>Model</label>
          <p className={fieldHintClass}>Required. Missing model selection breaks summary and chat generation.</p>
        </div>
        <input
          aria-label="Model"
          className={`${settingsInputClass} ${settingsControlClass}`}
          id={modelId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({
              ...current,
              model: event.target.value,
            }))
          }
          value={localAi.model}
        />
      </div>

      <div className={settingsFieldClass}>
        <div className={settingsFieldMetaClass}>
          <label className={fieldLabelClass} htmlFor={apiKeyId}>API Key</label>
          <p className={fieldHintClass}>Required for hosted providers. Missing credentials break remote inference.</p>
        </div>
        <input
          aria-label="API Key"
          className={`${settingsInputClass} ${settingsControlClass}`}
          id={apiKeyId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({
              ...current,
              api_key: event.target.value,
            }))
          }
          type="password"
          value={localAi.api_key}
        />
      </div>

      <div className={settingsFieldClass}>
        <div className={settingsFieldMetaClass}>
          <label className={fieldLabelClass} htmlFor={baseUrlId}>Base URL</label>
          <p className={fieldHintClass}>{baseUrlMeta.hint}</p>
        </div>
        <input
          aria-label="Base URL"
          className={`${settingsInputClass} ${settingsControlClass}`}
          id={baseUrlId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({
              ...current,
              base_url: event.target.value,
            }))
          }
          required
          placeholder={baseUrlMeta.placeholder}
          value={localAi.base_url}
        />
      </div>
    </div>
  );
}
