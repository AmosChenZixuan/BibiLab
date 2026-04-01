import { useEffect, useState } from "react";
import { FiSettings, FiUser } from "react-icons/fi";
import { MdTranslate } from "react-icons/md";
import { NavLink, Outlet } from "react-router-dom";

import { useLanguage } from "../../app/LanguageContext";
import { JobsBadge } from "../jobs/JobsBadge";
import { api } from "../../lib/api";
import type { HealthResponse } from "../../lib/types";
import IdentityPanel from "./IdentityPanel";

type HealthTier = "operational" | "degraded" | "unavailable";

function deriveHealthTier(health: HealthResponse): HealthTier {
  if (health.overall === "error") {
    return "unavailable";
  }

  if (
    health.dependencies.cuda?.status !== "ok" ||
    health.dependencies.embedding_model?.status !== "ok"
  ) {
    return "degraded";
  }

  return "operational";
}

const HEALTH_META: Record<HealthTier, { label: string; className: string }> = {
  operational: { label: "Operational", className: "bg-[#4ade80]" },
  degraded: { label: "Degraded", className: "bg-[#fbbf24]" },
  unavailable: { label: "Unavailable", className: "bg-[#f87171]" },
};

export function AppFrame() {
  const [healthTier, setHealthTier] = useState<HealthTier>("operational");
  const [identityOpen, setIdentityOpen] = useState(false);
  const { lang, setLang } = useLanguage();

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const next = await api.getHealth();
        if (!cancelled) {
          setHealthTier(deriveHealthTier(next));
        }
      } catch {
        if (!cancelled) {
          setHealthTier("unavailable");
        }
      }
    }

    void loadHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  const healthMeta = HEALTH_META[healthTier];

  return (
    <>
      <nav className="fixed inset-x-0 top-0 z-[100] flex h-[52px] items-center justify-between border-b border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.9)] px-[clamp(16px,3vw,48px)] shadow-[0_12px_28px_rgba(116,148,194,0.08)] backdrop-blur-[18px]">
        <NavLink className="inline-flex items-center" to="/" aria-label="Home">
          <span className='inline-flex h-7 w-7 items-center justify-center rounded-lg bg-[linear-gradient(135deg,#f3a9c9_0%,#f58bb9_58%,#a9e7ff_100%)] font-["Iowan_Old_Style","Palatino_Linotype",serif] text-base font-bold text-[#fff9f4]'>
            L
          </span>
        </NavLink>

        <div className="inline-flex items-center gap-2">
          <NavLink
            to="/settings"
            className="relative inline-flex h-9 w-9 items-center justify-center rounded-[10px] transition hover:bg-[rgba(125,217,255,0.12)]"
            title={healthMeta.label}
            aria-label="Settings"
          >
            <span className="inline-flex h-[18px] w-[18px] items-center justify-center text-[#5f7b9f]" aria-hidden="true">
              <FiSettings className="h-[18px] w-[18px]" />
            </span>
            <span
              className={`absolute right-1 bottom-1 h-[9px] w-[9px] rounded-full border-2 border-[rgba(255,255,255,0.92)] ${healthMeta.className}`}
            />
          </NavLink>
          <JobsBadge />

          <button
            type="button"
            className="relative inline-flex h-9 w-9 items-center justify-center rounded-[10px] bg-transparent text-[#274970] transition hover:bg-[rgba(125,217,255,0.12)]"
            aria-label={`Language: ${lang === "en" ? "English" : "Chinese"}`}
            title={lang === "en" ? "English" : "Chinese"}
            onClick={() => setLang(lang === "en" ? "zh" : "en")}
          >
            <span className="inline-flex h-[18px] w-[18px] items-center justify-center text-[#5f7b9f]" aria-hidden="true">
              <MdTranslate className="h-[18px] w-[18px]" />
            </span>
            <span
              className="absolute right-[2px] bottom-[2px] inline-flex h-[14px] min-w-[14px] items-center justify-center rounded-full border-2 border-[rgba(255,255,255,0.92)] bg-[rgba(255,255,255,0.96)] px-[3px] text-[0.5rem] leading-none font-bold text-[#5f7b9f]"
              aria-hidden="true"
            >
              {lang === "en" ? "EN" : "中"}
            </span>
          </button>

          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-[10px] bg-transparent text-[#274970] transition hover:bg-[rgba(125,217,255,0.12)]"
            aria-label="Identity"
            aria-expanded={identityOpen}
            aria-haspopup="menu"
            onClick={() => setIdentityOpen((open) => !open)}
          >
            <span className="inline-flex h-[18px] w-[18px] items-center justify-center text-[#5f7b9f]" aria-hidden="true">
              <FiUser className="h-[18px] w-[18px]" />
            </span>
          </button>
        </div>

        {identityOpen ? <IdentityPanel onClose={() => setIdentityOpen(false)} /> : null}
      </nav>

      <div className="min-h-screen px-[clamp(16px,3vw,48px)] pt-[calc(52px+28px)] pb-6 max-[820px]:px-3 max-[820px]:pt-[calc(52px+18px)] max-[820px]:pb-3">
        <main className="w-full rounded-[28px] border border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.72)] px-7 pt-8 pb-7 shadow-[0_18px_44px_rgba(102,136,181,0.08)] backdrop-blur-[18px] max-[820px]:rounded-[20px] max-[820px]:p-[18px]">
          <Outlet />
        </main>
      </div>
    </>
  );
}
