import { useEffect, useId, useState } from "react";

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
  const baseUrlMeta = BASE_URL_META[localAi.protocol] ?? BASE_URL_META.anthropic;
  const providerId = useId();
  const modelId = useId();
  const apiKeyId = useId();
  const baseUrlId = useId();
  const outputLanguageId = useId();

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
    </div>
  );
}
