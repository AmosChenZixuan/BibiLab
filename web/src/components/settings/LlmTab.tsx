import { useEffect, useId, useState } from "react";

import type { LocusConfig } from "../../lib/types";
import { Input, SettingsField } from "../../components/ui";

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
      <SettingsField
        label="Provider"
        hint="Required. Without a provider, LLM requests cannot start."
        htmlFor={providerId}
      >
        <select
          aria-label="Provider"
          className="w-full rounded-xl border border-border bg-white/92 px-3 py-2.5 text-ink outline-none transition focus:border-blue/45 focus:ring-2 focus:ring-sky/18 h-11 min-h-11"
          id={providerId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({ ...current, provider: event.target.value }))
          }
          value={localAi.provider}
        >
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
          <option value="ollama">Ollama</option>
          <option value="custom">Custom</option>
        </select>
      </SettingsField>

      <SettingsField
        label="Model"
        hint="Required. Missing model selection breaks summary and chat generation."
        htmlFor={modelId}
      >
        <Input
          aria-label="Model"
          id={modelId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({ ...current, model: event.target.value }))
          }
          inputSize="sm"
          value={localAi.model}
        />
      </SettingsField>

      <SettingsField
        label="API Key"
        hint="Required for hosted providers. Missing credentials break remote inference."
        htmlFor={apiKeyId}
      >
        <Input
          aria-label="API Key"
          id={apiKeyId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({ ...current, api_key: event.target.value }))
          }
          inputSize="sm"
          type="password"
          value={localAi.api_key}
        />
      </SettingsField>

      <SettingsField
        label="Base URL"
        hint={baseUrlMeta.hint}
        htmlFor={baseUrlId}
      >
        <Input
          aria-label="Base URL"
          id={baseUrlId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({ ...current, base_url: event.target.value }))
          }
          placeholder={baseUrlMeta.placeholder}
          required
          inputSize="sm"
          value={localAi.base_url}
        />
      </SettingsField>
    </div>
  );
}
