import { useEffect, useState } from "react";
import { FiSettings, FiUser } from "react-icons/fi";
import { MdTranslate } from "react-icons/md";
import { NavLink, Outlet } from "react-router-dom";

import { useLanguage } from "../../app/LanguageContext";
import { JobActivityProvider } from "../jobs/JobActivityProvider";
import { JobSpirit } from "../jobs/JobSpirit";
import { api, HEALTH_REFRESH_EVENT } from "../../lib/api";
import { deriveOverallHealthTier, HEALTH_META } from "../../lib/health";
import type { HealthResponse } from "../../lib/types";
import IdentityPanel from "./IdentityPanel";

export function AppFrame() {
  const [healthTier, setHealthTier] = useState<keyof typeof HEALTH_META>("operational");
  const [identityOpen, setIdentityOpen] = useState(false);
  const { lang, setLang } = useLanguage();

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const next = await api.getHealth();
        if (!cancelled) {
          setHealthTier(deriveOverallHealthTier(next));
        }
      } catch {
        if (!cancelled) {
          setHealthTier("unavailable");
        }
      }
    }

    function handleHealthRefresh(event: Event) {
      const next = (event as CustomEvent<HealthResponse>).detail;
      setHealthTier(deriveOverallHealthTier(next));
    }

    void loadHealth();
    window.addEventListener(HEALTH_REFRESH_EVENT, handleHealthRefresh);
    return () => {
      cancelled = true;
      window.removeEventListener(HEALTH_REFRESH_EVENT, handleHealthRefresh);
    };
  }, []);

  const healthMeta = HEALTH_META[healthTier];

  return (
    <JobActivityProvider>
      <nav className="fixed inset-x-0 top-0 z-nav flex h-[52px] items-center justify-between px-[clamp(16px,3vw,48px)] /* fluid navbar padding */ bg-white">
        <NavLink className="inline-flex items-center" to="/" aria-label="Home">
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-[linear-gradient(135deg,#f3a9c9_0%,#f58bb9_58%,#a9e7ff_100%)] font-serif text-base font-bold text-[#fff9f4]">
            L
          </span>
        </NavLink>

        <div className="inline-flex items-center gap-2">
          <NavLink
            to="/settings"
            className="relative inline-flex h-9 w-9 items-center justify-center rounded-icon transition hover:bg-sky/12"
            title={healthMeta.label}
            aria-label="Settings"
          >
            <span className="inline-flex h-[18px] w-[18px] items-center justify-center text-blue/70" aria-hidden="true">
              <FiSettings className="h-[18px] w-[18px]" />
            </span>
            <span
              className={`absolute right-1 bottom-1 h-[9px] w-[9px] rounded-full border-2 border-white/92 ${healthMeta.className}`}
            />
          </NavLink>

          <button
            type="button"
            className="relative inline-flex h-9 w-9 items-center justify-center rounded-icon bg-transparent text-ink transition hover:bg-sky/12"
            aria-label={`Language: ${lang === "en" ? "English" : "Chinese"}`}
            title={lang === "en" ? "English" : "Chinese"}
            onClick={() => setLang(lang === "en" ? "zh" : "en")}
          >
            <span className="inline-flex h-[18px] w-[18px] items-center justify-center text-blue/70" aria-hidden="true">
              <MdTranslate className="h-[18px] w-[18px]" />
            </span>
            <span
              className="absolute right-0.5 bottom-0.5 inline-flex h-3.5 min-w-3.5 items-center justify-center rounded-full border-2 border-white/92 bg-white/96 px-[3px] text-badge leading-none font-bold text-blue/70"
              aria-hidden="true"
            >
              {lang === "en" ? "EN" : "中"}
            </span>
          </button>

          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-icon bg-transparent text-ink transition hover:bg-sky/12"
            aria-label="Identity"
            aria-expanded={identityOpen}
            aria-haspopup="menu"
            onClick={() => setIdentityOpen((open) => !open)}
          >
            <span className="inline-flex h-[18px] w-[18px] items-center justify-center text-blue/70" aria-hidden="true">
              <FiUser className="h-[18px] w-[18px]" />
            </span>
          </button>
        </div>

        {identityOpen ? <IdentityPanel onClose={() => setIdentityOpen(false)} /> : null}
      </nav>

      <div className="min-h-screen px-[clamp(16px,10vw,160px)] /* fluid content padding: clamps 16px..160px */ pt-[calc(52px+28px)] /* 52px nav + 28px breathing room */ pb-6 max-[820px]:px-3 max-[820px]:pt-[calc(52px+18px)] max-[820px]:pb-3">
        <main>
          <Outlet />
        </main>
      </div>
      <JobSpirit />
    </JobActivityProvider>
  );
}
