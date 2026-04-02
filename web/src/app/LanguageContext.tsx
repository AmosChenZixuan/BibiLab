import { createContext, useContext, useState, type ReactNode } from "react";

type Lang = "en" | "zh";

type LanguageContextValue = {
  lang: Lang;
  setLang: (lang: Lang) => void;
};

const LanguageContext = createContext<LanguageContextValue>({
  lang: "en",
  setLang: () => {},
});

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => {
    const stored = localStorage.getItem("locus-lang");
    return stored === "zh" ? "zh" : "en";
  });

  function setLang(nextLang: Lang) {
    setLangState(nextLang);
    localStorage.setItem("locus-lang", nextLang);
  }

  return <LanguageContext.Provider value={{ lang, setLang }}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  return useContext(LanguageContext);
}
