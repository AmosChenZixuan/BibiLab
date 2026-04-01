import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { api } from "../../lib/api";
import type { HealthResponse } from "../../lib/types";

export function AppFrame() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const next = await api.getHealth();
        if (!cancelled) {
          setHealth(next);
        }
      } catch {
        if (!cancelled) {
          setHealth({
            overall: "error",
            dependencies: {},
          });
        }
      }
    }

    void loadHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app-shell">
      <header className="floating-nav">
        <NavLink className="brand-mark" to="/">
          <span className="brand-mark__glyph">L</span>
          <span>
            <span className="brand-mark__title">Locus</span>
            <span className="brand-mark__subtitle">Private video notebooks</span>
          </span>
        </NavLink>
        <div className="floating-nav__right">
          <div className={`health-pill health-pill--${health?.overall ?? "loading"}`}>
            <span className="health-pill__dot" />
            <span>
              {health?.overall === "ok"
                ? "System healthy"
                : health?.overall === "error"
                  ? "System needs attention"
                  : "Checking system"}
            </span>
          </div>
          <NavLink className={({ isActive }) => `nav-link${isActive ? " active" : ""}`} to="/settings">
            Settings
          </NavLink>
        </div>
      </header>
      <div className="app-frame">
        <main className="page">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
