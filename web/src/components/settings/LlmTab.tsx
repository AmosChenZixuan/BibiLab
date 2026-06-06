import { useEffect, useId, useState } from "react";

import { Eye, EyeOff } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import type { BibilabConfig, OutputLanguage } from "@/lib/types";
import { Input, Select, SettingsField } from "@/components/ui";

type LlmTabProps = {
  config: BibilabConfig;
  onBlur: (updated: BibilabConfig) => void;
};

const BASE_URL_META: Record<string, { hintKey: string; placeholderKey: string }> = {
  openai: { hintKey: "settings.openaiBaseUrlHint", placeholderKey: "settings.openaiBaseUrlPlaceholder" },
  anthropic: { hintKey: "settings.anthropicBaseUrlHint", placeholderKey: "settings.anthropicBaseUrlPlaceholder" },
};

export function LlmTab({ config, onBlur }: LlmTabProps) {
  const { t } = useLanguage();
  const [localAi, setLocalAi] = useState(config.ai);
  const [showApiKey, setShowApiKey] = useState(false);
  const baseUrlMeta = BASE_URL_META[localAi.protocol] ?? BASE_URL_META.openai;
  const providerId = useId();
  const modelId = useId();
  const apiKeyId = useId();
  const baseUrlId = useId();
  const outputLanguageId = useId();
  const contextWindowId = useId();

  useEffect(() => {
    setLocalAi(config.ai);
  }, [config.ai]);

  function handleBlur() {
    onBlur({ ...config, ai: localAi });
  }

  return (
    <div className="grid gap-3">
      <SettingsField
        label={t("settings.provider")}
        hint={t("settings.providerRequired")}
        htmlFor={providerId}
      >
        <Select
          aria-label="Provider"
          id={providerId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({ ...current, protocol: event.target.value }))
          }
          value={localAi.protocol}
        >
          <option value="openai">{t("settings.protocolOpenai")}</option>
          <option value="anthropic">{t("settings.protocolAnthropic")}</option>
        </Select>
      </SettingsField>

      <SettingsField
        label={t("settings.model")}
        hint={t("settings.modelRequired")}
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
        label={t("settings.apiKey")}
        hint={t("settings.apiKeyRequired")}
        htmlFor={apiKeyId}
      >
        <div className="relative">
          <Input
            aria-label="API Key"
            id={apiKeyId}
            onBlur={handleBlur}
            onChange={(event) =>
              setLocalAi((current) => ({ ...current, api_key: event.target.value }))
            }
            inputSize="sm"
            type={showApiKey ? "text" : "password"}
            value={localAi.api_key}
            className="pr-9"
          />
          <button
            type="button"
            aria-label={showApiKey ? "Hide API key" : "Reveal API key"}
            className="absolute right-0 top-0 flex h-full items-center justify-center px-3 text-ink/40 hover:text-ink"
            onClick={() => setShowApiKey((v) => !v)}
          >
            {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        </div>
      </SettingsField>

      <SettingsField
        label={t("settings.baseUrl")}
        hint={t(baseUrlMeta.hintKey)}
        htmlFor={baseUrlId}
      >
        <Input
          aria-label="Base URL"
          id={baseUrlId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({ ...current, base_url: event.target.value }))
          }
          placeholder={t(baseUrlMeta.placeholderKey)}
          required
          inputSize="sm"
          value={localAi.base_url}
        />
      </SettingsField>

      <SettingsField
        label={t("settings.outputLanguage")}
        hint={t("settings.outputLanguageDesc")}
        htmlFor={outputLanguageId}
      >
        <Select
          aria-label="Output Language"
          id={outputLanguageId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({ ...current, output_language: event.target.value as OutputLanguage }))
          }
          value={localAi.output_language ?? "ui"}
        >
          <option value="ui">{t("settings.outputLanguageFollowUi")}</option>
          <option value="en">{t("settings.outputLanguageEnglish")}</option>
          <option value="zh">{t("settings.outputLanguageChinese")}</option>
        </Select>
      </SettingsField>

      <SettingsField
        label={t("settings.contextWindow")}
        hint={t("settings.contextWindowDesc")}
        htmlFor={contextWindowId}
      >
        <Select
          aria-label={t("settings.contextWindow")}
          id={contextWindowId}
          onBlur={handleBlur}
          onChange={(event) =>
            setLocalAi((current) => ({ ...current, context_window: Number(event.target.value) }))
          }
          value={localAi.context_window}
        >
          <option value={64000}>64K</option>
          <option value={128000}>128K</option>
          <option value={200000}>200K</option>
          <option value={1000000}>1M</option>
        </Select>
      </SettingsField>
    </div>
  );
}
