import { useEffect, useState } from "react";
import { Settings, User } from "lucide-react";
import { Languages } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

import { BilibiliQrModal } from "@/components/auth/BilibiliQrModal";
import { useLanguage } from "@/app/LanguageContext";
import { JobActivityProvider } from "@/components/jobs/JobActivityProvider";
import { JobSpirit } from "@/components/jobs/JobSpirit";
import { api, HEALTH_REFRESH_EVENT, BILIBILI_AUTH_REFRESH_EVENT, notifyBilibiliAuthChanged } from "@/lib/api";
import { deriveOverallHealthTier, HEALTH_META } from "@/lib/health";
import type { HealthResponse } from "@/lib/types";
import { NavTitleContext } from "./NavTitleContext";
import IdentityPanel from "./IdentityPanel";

export function AppFrame() {
  const [healthTier, setHealthTier] = useState<keyof typeof HEALTH_META>("healthy");
  const [identityOpen, setIdentityOpen] = useState(false);
  const [bilibiliCookie, setBilibiliCookie] = useState("");
  const [qrModalOpen, setQrModalOpen] = useState(false);
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

  useEffect(() => {
    let cancelled = false;

    async function loadConfig() {
      try {
        const config = await api.getConfig();
        if (!cancelled && config) {
          setBilibiliCookie(config.accounts.bilibili.cookie);
        }
      } catch {
        // config fetch failure is non-critical for navbar display
      }
    }

    void loadConfig();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    async function refreshCookie() {
      try {
        const config = await api.getConfig();
        if (config) {
          setBilibiliCookie(config.accounts.bilibili.cookie);
        }
      } catch {
        // non-critical
      }
    }

    function handleAuthRefresh() {
      void refreshCookie();
    }

    window.addEventListener(BILIBILI_AUTH_REFRESH_EVENT, handleAuthRefresh);
    return () => {
      window.removeEventListener(BILIBILI_AUTH_REFRESH_EVENT, handleAuthRefresh);
    };
  }, []);

  async function handleLoginSuccess() {
    try {
      const config = await api.getConfig();
      if (config) {
        setBilibiliCookie(config.accounts.bilibili.cookie);
      }
    } catch {
      // refresh failure is non-critical
    }
    notifyBilibiliAuthChanged();
    setQrModalOpen(false);
  }

  async function handleLogout() {
    try {
      await api.auth.deleteBilibiliAuth();
      const config = await api.getConfig();
      if (config) {
        setBilibiliCookie(config.accounts.bilibili.cookie);
      }
    } catch {
      // logout failure is non-critical
    }
    notifyBilibiliAuthChanged();
  }

  const healthMeta = HEALTH_META[healthTier];
  const { t } = useLanguage();

  return (
    <JobActivityProvider>
      <NavTitleContext.Provider value={navElement}>
        <nav
          ref={setNavElement}
          className="fixed inset-x-0 top-0 z-nav flex h-14 items-center justify-between bg-white px-4 md:px-6 xl:px-12"
        >
          <NavLink className="inline-flex items-center gap-2" to="/" aria-label="Home">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-pink text-base font-bold text-white/95">
              B
            </span>
          </NavLink>

          <div className="inline-flex items-center gap-2">
            <NavLink
              to="/settings"
              className="relative inline-flex h-9 w-9 items-center justify-center rounded-lg transition hover:bg-sky/10"
              title={t("health." + healthTier)}
              aria-label="Settings"
            >
              <span className="inline-flex size-4.5 items-center justify-center text-blue/70" aria-hidden="true">
                <Settings className="size-4.5" />
              </span>
              <span
                className={`absolute bottom-1 right-1 size-2 rounded-full border-2 border-white/92 ${healthMeta.className}`}
              />
            </NavLink>

            <button
              type="button"
              className="relative inline-flex h-9 w-9 items-center justify-center rounded-lg bg-transparent text-ink transition hover:bg-sky/10"
              aria-label="Language"
              title={t(lang === "en" ? "navbar.languageEn" : "navbar.languageZh")}
              onClick={() => setLang(lang === "en" ? "zh" : "en")}
            >
              <span className="inline-flex size-4.5 items-center justify-center text-blue/70" aria-hidden="true">
                <Languages className="size-4.5" />
              </span>
              <span
                className="absolute bottom-0.5 right-0.5 inline-flex h-3.5 min-w-3.5 items-center justify-center rounded-full border-2 border-white/92 bg-white/96 px-1 text-xs leading-none font-bold text-blue/70"
                aria-hidden="true"
              >
                {lang === "en" ? "EN" : "中"}
              </span>
            </button>

            <button
              type="button"
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-transparent text-ink transition hover:bg-sky/10"
              aria-label="Identity"
              aria-expanded={identityOpen}
              aria-haspopup="menu"
              onClick={() => setIdentityOpen((open) => !open)}
            >
              <span className="inline-flex size-4.5 items-center justify-center text-blue/70" aria-hidden="true">
                <User className="size-4.5" />
              </span>
            </button>
          </div>

          {identityOpen ? (
            <IdentityPanel
              bilibiliCookie={bilibiliCookie}
              onClose={() => setIdentityOpen(false)}
              onLogin={() => setQrModalOpen(true)}
              onLogout={handleLogout}
            />
          ) : null}
        </nav>

        <BilibiliQrModal
          open={qrModalOpen}
          onClose={() => setQrModalOpen(false)}
          onSuccess={handleLoginSuccess}
        />

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
