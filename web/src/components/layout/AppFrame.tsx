import { useEffect, useState } from "react";
import { Settings, User } from "lucide-react";
import { Languages } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

import { useLanguage } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { JobSpirit } from "@/components/jobs/JobSpirit";
import { api, HEALTH_REFRESH_EVENT } from "@/lib/api";
import { deriveOverallHealthTier, HEALTH_META } from "@/lib/health";
import type { HealthResponse } from "@/lib/types";
import { NavTitleContext } from "./NavTitleContext";
import IdentityPanel from "./IdentityPanel";

export function AppFrame() {
  const [healthTier, setHealthTier] = useState<keyof typeof HEALTH_META>("healthy");
  const [identityOpen, setIdentityOpen] = useState(false);
  const { lang, setLang } = useLanguage();
  const [navElement, setNavElement] = useState<HTMLElement | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const next = await api.getHealth();
        if (!cancelled && next) {
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
  const { t } = useLanguage();

  return (
    <JobActivityProvider>
      <NavTitleContext.Provider value={navElement}>
        <nav
          ref={setNavElement}
          className="fixed inset-x-0 top-0 z-nav flex h-14 items-center justify-between glass border-b border-black/10 px-4 md:px-6 xl:px-12"
        >
          <NavLink className="inline-flex items-center gap-2" to="/" aria-label="Home">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-pink text-body-compact font-bold text-white/95">
              B
            </span>
          </NavLink>

          <div className="inline-flex items-center gap-2">
            <NavLink
              to="/settings"
              className="relative inline-flex h-9 w-9 items-center justify-center rounded-lg transition hover:bg-sky-blue-light"
              title={t("health." + healthTier)}
              aria-label="Settings"
            >
              <span className="inline-flex size-4.5 items-center justify-center text-sky-blue/70" aria-hidden="true">
                <Settings className="size-4.5" />
              </span>
              <span
                className={`absolute bottom-1 right-1 size-2 rounded-full border-2 border-white/92 ${healthMeta.className}`}
              />
            </NavLink>

            <button
              type="button"
              className="relative inline-flex h-9 w-9 items-center justify-center rounded-lg bg-transparent text-charcoal transition hover:bg-sky-blue-light"
              aria-label="Language"
              title={t(lang === "en" ? "navbar.languageEn" : "navbar.languageZh")}
              onClick={() => setLang(lang === "en" ? "zh" : "en")}
            >
              <span className="inline-flex size-4.5 items-center justify-center text-sky-blue/70" aria-hidden="true">
                <Languages className="size-4.5" />
              </span>
              <span
                className="absolute bottom-0.5 right-0.5 inline-flex h-3.5 min-w-3.5 items-center justify-center rounded-full border-2 border-white/92 bg-white/96 px-1 text-small leading-none font-bold text-sky-blue/70"
                aria-hidden="true"
              >
                {lang === "en" ? "EN" : "中"}
              </span>
            </button>

            <button
              type="button"
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-transparent text-charcoal transition hover:bg-sky-blue-light"
              aria-label="Identity"
              aria-expanded={identityOpen}
              aria-haspopup="menu"
              onClick={() => setIdentityOpen((open) => !open)}
            >
              <span className="inline-flex size-4.5 items-center justify-center text-sky-blue/70" aria-hidden="true">
                <User className="size-4.5" />
              </span>
            </button>
          </div>

          {identityOpen ? <IdentityPanel onClose={() => setIdentityOpen(false)} /> : null}
        </nav>

        <div className="min-h-screen px-4 pb-6 pt-20 md:px-10 md:pt-24 xl:px-24 2xl:px-40">
          <main>
            <Outlet />
          </main>
        </div>
        <JobSpirit />
      </NavTitleContext.Provider>
    </JobActivityProvider>
  );
}
