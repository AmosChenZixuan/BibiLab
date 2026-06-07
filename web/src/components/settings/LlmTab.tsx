import { useEffect, useId, useState } from "react";

import { Eye, EyeOff } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import type { BibilabConfig, OutputLanguage } from "@/lib/types";
import { Input, Select, SettingsField, SlotSlider } from "@/components/ui";
import type { SlotOption } from "@/components/ui/SlotSlider";

type LlmTabProps = {
  config: BibilabConfig;
  onBlur: (updated: BibilabConfig) => void;
};

const BASE_URL_META: Record<string, { hintKey: string; placeholderKey: string }> = {
  openai: { hintKey: "settings.openaiBaseUrlHint", placeholderKey: "settings.openaiBaseUrlPlaceholder" },
  anthropic: { hintKey: "settings.anthropicBaseUrlHint", placeholderKey: "settings.anthropicBaseUrlPlaceholder" },
};

// 4 user-selectable model context windows. These bound the input-side
// fail-loud threshold in resolve_max_tokens; the LLM call will fail-loud
// (with the localized "Input exceeds model context window" message) when
// input + max_output + margin > one of these values.
const CONTEXT_WINDOW_OPTIONS: SlotOption<number>[] = [
  { value: 64000, label: "64K" },
  { value: 128000, label: "128K" },
  { value: 200000, label: "200K" },
  { value: 1000000, label: "1M" },
];

// Per-call output budget. User-chosen from a slot slider so each user can
// pick what fits their model. Default 16K matches the observed upper bound
// on thinking + answer for all real tasks.
const MAX_OUTPUT_TOKENS_OPTIONS: SlotOption<number>[] = [
  { value: 16384, label: "16K" },
  { value: 32768, label: "32K" },
  { value: 65536, label: "64K" },
  { value: 102400, label: "100K" },
];

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
  const maxOutputTokensId = useId();

  useEffect(() => {
    setLocalAi(config.ai);
  }, [config.ai]);

  function commitChange(ai: typeof localAi) {
    onBlur({ ...config, ai });
  }

  function handleBlur() {
    commitChange(localAi);
  }

  // Update local state for every selection (so the inline alert reflects the
  // chosen value), but only persist a pair the backend validator accepts:
  // max_output_tokens must be strictly less than context_window. Committing an
  // invalid pair would just bounce off the pydantic model_validator with a 422.
  function applyAi(next: typeof localAi) {
    setLocalAi(next);
    if (next.max_output_tokens < next.context_window) {
      commitChange(next);
    }
  }

  // Cross-field guard: max_output_tokens must be strictly less than
  // context_window. Surfaced inline so the user sees the issue immediately
  // and can fix it before save.
  const budgetInvalid =
    typeof localAi.max_output_tokens === "number" &&
    typeof localAi.context_window === "number" &&
    localAi.max_output_tokens >= localAi.context_window;

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
        <SlotSlider
          ariaLabel={t("settings.contextWindow")}
          id={contextWindowId}
          onChange={(value) => applyAi({ ...localAi, context_window: value })}
          options={CONTEXT_WINDOW_OPTIONS}
          value={localAi.context_window}
        />
      </SettingsField>

      <SettingsField
        label={t("settings.maxOutputTokens")}
        hint={t("settings.maxOutputTokensDesc")}
        htmlFor={maxOutputTokensId}
      >
        <SlotSlider
          ariaLabel={t("settings.maxOutputTokens")}
          id={maxOutputTokensId}
          onChange={(value) => applyAi({ ...localAi, max_output_tokens: value })}
          options={MAX_OUTPUT_TOKENS_OPTIONS}
          value={localAi.max_output_tokens}
        />
        {budgetInvalid ? (
          <p
            role="alert"
            className="mt-2 text-sm text-red"
            data-testid="max-output-error"
          >
            {t("settings.maxOutputTokensTooLarge")}
          </p>
        ) : null}
      </SettingsField>
    </div>
  );
}
