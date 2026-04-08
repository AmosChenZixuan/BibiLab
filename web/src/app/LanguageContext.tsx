import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

import en from "@/lib/i18n/en.json";
import zh from "@/lib/i18n/zh.json";

type Lang = "en" | "zh";

type LanguageContextValue = {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
};

const dictionaries: Record<Lang, Record<string, unknown>> = { en, zh };

function getNestedValue(obj: Record<string, unknown>, path: string): string {
  const keys = path.split(".");
  let result: unknown = obj;
  for (const key of keys) {
    if (result && typeof result === "object" && key in result) {
      result = (result as Record<string, unknown>)[key];
    } else {
      return path;
    }
  }
  return typeof result === "string" ? result : path;
}

const INTERPOLATE_RE = /%\{(\w+)\}/g;

function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) {
    return template;
  }
  return template.replace(INTERPOLATE_RE, (_, key) => String(params[key] ?? "%{" + key + "}"));
}

const LanguageContext = createContext<LanguageContextValue>({
  lang: "en",
  setLang: () => {},
  t: (key: string, params?: Record<string, string | number>) => key,
});

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => {
    const stored = localStorage.getItem("bibilab-lang");
    return stored === "zh" ? "zh" : "en";
  });

  function setLang(nextLang: Lang) {
    setLangState(nextLang);
    localStorage.setItem("bibilab-lang", nextLang);
  }

  const t = useCallback(
    (key: string, params?: Record<string, string | number>): string => {
      const value = getNestedValue(dictionaries[lang], key);
      return interpolate(value, params);
    },
    [lang],
  );

  return <LanguageContext.Provider value={{ lang, setLang, t }}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  return useContext(LanguageContext);
}
