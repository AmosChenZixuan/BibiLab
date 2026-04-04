import { createContext, useContext, useState, type ReactNode } from "react";

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

function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) {
    return template;
  }
  return template.replace(/%\{(\w+)\}/g, (_, key) => String(params[key] ?? "%{" + key + "}"));
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

  function t(key: string, params?: Record<string, string | number>): string {
    const value = getNestedValue(dictionaries[lang], key);
    return interpolate(value, params);
  }

  return <LanguageContext.Provider value={{ lang, setLang, t }}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  return useContext(LanguageContext);
}
