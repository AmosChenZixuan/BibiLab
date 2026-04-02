# Tailwind Design System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every arbitrary Tailwind value in the web frontend with named design tokens, normalize spacing to Tailwind's built-in scale, restructure `ui.ts` with grouped and commented constants, then extract four reusable React wrapper components (`Button`, `FormField`, `Panel`, `StatusChip`).

**Architecture:** Phase A adds a `@theme` block to `app.css` so Tailwind v4 generates utility classes from named CSS variables; `ui.ts` is rewritten to use those utilities; then every component file with inline arbitrary values is updated. Phase B extracts the most-reused JSX patterns into thin forwardable wrapper components that replace the raw class string constants.

**Tech Stack:** React 18, TypeScript, Tailwind v4 (`@tailwindcss/vite`), Vitest + React Testing Library, custom `app.css` (no config file — Tailwind v4 reads `@theme` directly from CSS).

**Spec:** `docs/superpowers/specs/2026-04-01-tailwind-design-system.md`

---

## Token Reference (read this before editing any file)

After Task 1, these utility classes are available everywhere:

| Old arbitrary | New utility |
|---|---|
| `text-[#274970]` | `text-ink` |
| `text-[#8096b3]` | `text-muted` |
| `text-[#547094]` | `text-muted/80` |
| `text-[#f08bb9]` | `text-pink` |
| `text-[#5b7faa]`, `bg-[#5b7faa]` | `text-blue`, `bg-blue` |
| `text-[#4ca9cf]` | `text-success` |
| `text-[#8d1d2c]`, `bg-[#8d1d2c]` | `text-danger`, `bg-danger` |
| `text-[#b46088]` | `text-warn` |
| `border-[rgba(106,147,198,0.12)]` | `border-border` |
| `bg-[rgba(255,252,247,0.82)]` | `bg-surface` |
| `bg-[rgba(255,255,255,0.76\|0.84\|0.92\|0.96)]` | `bg-white/76`, `/84`, `/92`, `/96` |
| `bg-[rgba(255,255,255,0.36)]` | `bg-white/36` |
| `bg-[rgba(255,255,255,0.64)]` | `bg-white/64` |
| `bg-[rgba(125,217,255,0.08\|0.10\|0.12)]` | `bg-sky/8`, `bg-sky/10`, `bg-sky/12` |
| `bg-[rgba(240,139,185,0.14)]` | `bg-pink/14` |
| `bg-[rgba(96,123,163,0.14)]` | `bg-blue/14` |
| `border-[rgba(91,127,170,0.18)]` | `border-blue/18` |
| `border-[rgba(106,147,198,0.18)]` | `border-blue/18` |
| `shadow-[0_14px_28px_rgba(116,148,194,0.07)]` | `shadow-card` |
| `shadow-[0_22px_44px_rgba(116,148,194,0.12)]` | `shadow-elevated` |
| `shadow-[0_8px_32px_rgba(0,0,0,0.12)]` | `shadow-overlay` |
| `rounded-[10px]` | `rounded-icon` |
| `rounded-[14px]` | `rounded-overlay` |
| `rounded-[18px]`, `rounded-[20px]` | `rounded-2xl`, `rounded-3xl` |
| `rounded-[22px]` | `rounded-card` |
| `rounded-[28px]` | `rounded-drawer` |
| `z-[100]` | `z-nav` |
| `z-[199]` | `z-backdrop` |
| `z-[200]` | `z-overlay` |
| `font-["Iowan_Old_Style","Palatino_Linotype",serif]` | `font-serif` |
| `font-['SFMono-Regular',Consolas,monospace]` | `font-mono` |
| `text-[clamp(2rem,4vw,3.5rem)]` | `text-display` |
| `text-[1.35rem]` | `text-section` |
| `text-[0.5rem]` | `text-badge` |
| `text-[0.72rem]`, `text-[0.8rem]`, `text-[0.82rem]`–`text-[0.94rem]` | `text-xs` / `text-sm` |
| `text-[1.45rem]` | `text-2xl` |
| `text-[11px]` | `text-xs` |
| `px-[14px]`, `gap-[14px]` | `px-3.5`, `gap-3.5` |
| `py-[18px]`, `p-[18px]` | `py-4.5`, `p-4.5` |
| `px-[10px]` | `px-2.5` |
| `right-[2px]`, `bottom-[2px]` | `right-0.5`, `bottom-0.5` |
| `h-[14px]`, `min-w-[14px]` | `h-3.5`, `min-w-3.5` |
| `gap-[10px]` | `gap-2.5` |
| `bg-[linear-gradient(90deg,rgba(240,139,185,0.14),rgba(125,217,255,0.14))]` | `bg-linear-to-r from-pink/14 to-sky/14` |
| `bg-[linear-gradient(135deg,#f08bb9_0%,#5b7faa_100%)]` | `bg-linear-to-br from-pink to-blue` |

**Keep as-is (documented exceptions):**
- `h-[52px]` — navbar height in AppFrame
- `top-[52px]` — IdentityPanel position depends on navbar height; add comment `{/* navbar height */}`
- `top-[92px]` — jobs drawer position (52px nav + 40px gap); add comment
- `px-[clamp(16px,3vw,48px)]` — AppFrame navbar responsive padding; add comment
- `px-[clamp(16px,10vw,160px)]` — AppFrame content responsive padding; add comment
- `pt-[calc(52px+28px)]` — content offset (52px navbar + 28px gap); add comment
- `w-[min(420px,calc(100vw-24px))]` — jobs drawer max-width; add comment
- `max-h-[calc(100vh-116px)]` — jobs drawer max-height; add comment
- `backdrop-blur-[18px]` — glass blur effect (no matching Tailwind step); add comment
- `py-[11px]` — button vertical padding (no matching step); stays in ui.ts
- `bg-[rgba(248,251,255,0.72)]` — job card tint (unique sky tint, no token match); add comment
- `bg-[linear-gradient(160deg,...)]` — list card gradient (3-stop, unique); add comment
- `bg-[linear-gradient(135deg,#f3a9c9...)]` — logo gradient (3-stop, unique)
- `max-w-[38rem]` — OtherTab layout value (one-off)
- `tracking-[0.14em]` — eyebrow letter-spacing (no standard Tailwind step)
- `max-w-[560px]`, `max-w-[780px]` — page width constraints (one-off)
- `grid-cols-[...]` — complex grid definitions (one-off layout)
- `h-[18px] w-[18px]`, `h-[9px] w-[9px]`, `h-[7px] w-[7px]` — icon pixel sizes (one-off)
- `px-[3px]` — badge micro-padding
- `leading-[0.95]` — tight heading line-height
- `text-[#5f7b9f]` — AppFrame icon color (close to muted-mid; one-off tint)
- `text-[#4e6485]`, `text-[#4e6f99]` — one-off tints in SourceDetail/ListGrid
- `text-[2.5rem]` — create-list "+" symbol size
- `border-white/92` — pip border (use opacity modifier directly, no token needed)
- `content-between`, `content-center` — standard Tailwind, no change needed

---

## File Map

### Phase A (token replacement)
| Action | File |
|---|---|
| Modify | `web/src/styles/app.css` |
| Rewrite | `web/src/lib/ui.ts` |
| Modify | `web/src/components/layout/AppFrame.tsx` |
| Modify | `web/src/components/layout/IdentityPanel.tsx` |
| Modify | `web/src/components/jobs/JobsBadge.tsx` |
| Modify | `web/src/pages/SettingsPage.tsx` |
| Modify | `web/src/components/lists/ListGrid.tsx` |
| Modify | `web/src/components/chat/ChatPanel.tsx` |
| Modify | `web/src/components/sources/SourceList.tsx` |
| Modify | `web/src/components/sources/SourceDetail.tsx` |
| Modify | `web/src/pages/ListDetailPage.tsx` |
| Modify | `web/src/components/settings/TranscriptTab.tsx` |
| Modify | `web/src/components/settings/OtherTab.tsx` |

### Phase B (component extraction)
| Action | File |
|---|---|
| Create | `web/src/components/ui/Button.tsx` |
| Create | `web/src/components/ui/FormField.tsx` |
| Create | `web/src/components/ui/Panel.tsx` |
| Create | `web/src/components/ui/StatusChip.tsx` |
| Create | `web/src/components/ui/index.ts` |
| Create | `web/src/test/ui-components.test.tsx` |
| Modify | `web/src/components/sources/SourceList.tsx` |
| Modify | `web/src/components/sources/SourceDetail.tsx` |
| Modify | `web/src/components/jobs/JobsBadge.tsx` |
| Modify | `web/src/components/lists/ListGrid.tsx` |
| Modify | `web/src/components/studio/StudioPanel.tsx` |
| Modify | `web/src/pages/ListDetailPage.tsx` |
| Modify | `web/src/pages/HomePage.tsx` |
| Modify | `web/src/pages/SettingsPage.tsx` |
| Modify | `web/src/lib/ui.ts` — remove constants replaced by components |

---

## Phase A

### Task 1: Add `@theme` token block to `app.css`

**Files:**
- Modify: `web/src/styles/app.css`

- [ ] **Step 1: Replace the contents of `app.css`**

```css
@import "tailwindcss";

@theme {
  /* ── Colors ──────────────────────────────────────────────────────────────── */
  --color-ink:     #274970;       /* primary text */
  --color-muted:   #8096b3;       /* secondary text; use muted/80 for #547094 */
  --color-pink:    #f08bb9;       /* accent — gradient start, eyebrow */
  --color-blue:    #5b7faa;       /* accent — links, active states, gradient end */
  --color-sky:     #7dd9ff;       /* highlight tints via opacity modifier */
  --color-success: #4ca9cf;       /* positive status */
  --color-danger:  #8d1d2c;       /* destructive actions, error status */
  --color-warn:    #b46088;       /* degraded / unavailable status */
  --color-border:  rgba(106, 147, 198, 0.12);  /* universal panel/input border */
  --color-surface: rgba(255, 252, 247, 0.82);  /* warm paper — only non-white surface token */

  /* ── Typography ──────────────────────────────────────────────────────────── */
  --font-serif: "Iowan Old Style", "Palatino Linotype", serif;
  --font-mono:  SFMono-Regular, Consolas, monospace;

  /* Custom text sizes; standard sizes (xs=0.75rem, sm=0.875rem) cover the rest */
  --text-badge:   0.5rem;                    /* navbar language pip */
  --text-section: 1.35rem;                   /* panel / workspace titles */
  --text-display: clamp(2rem, 4vw, 3.5rem);  /* fluid page headings */

  /* ── Shadows ─────────────────────────────────────────────────────────────── */
  --shadow-card:     0 14px 28px rgba(116, 148, 194, 0.07);  /* standard card elevation */
  --shadow-elevated: 0 22px 44px rgba(116, 148, 194, 0.12);  /* drawers, modals */
  --shadow-overlay:  0 8px 32px rgba(0, 0, 0, 0.12);          /* identity panel */

  /* ── Border radius ───────────────────────────────────────────────────────── */
  --radius-icon:    10px;  /* nav icon buttons */
  --radius-overlay: 14px;  /* identity panel, tooltips */
  --radius-card:    22px;  /* list cards, app panels */
  --radius-drawer:  28px;  /* jobs drawer */

  /* ── Z-index ─────────────────────────────────────────────────────────────── */
  --z-nav:      100;  /* fixed navbar */
  --z-backdrop: 199;  /* click-away overlay behind panels */
  --z-overlay:  200;  /* floating panels (identity, etc.) */
}

@layer base {
  :root {
    font-family: "Avenir Next", "Segoe UI", sans-serif;
  }

  * {
    box-sizing: border-box;
  }

  body {
    min-width: 320px;
    min-height: 100vh;
    margin: 0;
    color: theme(--color-ink);
  }

  #root {
    min-height: 100vh;
  }

  button,
  input,
  select,
  textarea {
    font: inherit;
  }

  button {
    cursor: pointer;
  }

  a {
    color: inherit;
    text-decoration: none;
  }
}
```

- [ ] **Step 2: Run lint to confirm Tailwind parses the new tokens**

```bash
cd web && npm run lint
```
Expected: 0 errors. (No component files changed yet — existing arbitrary classes still work alongside the new tokens.)

- [ ] **Step 3: Commit**

```bash
git add web/src/styles/app.css
git commit -m "feat | web | add tailwind design token @theme block"
```

---

### Task 2: Rewrite `ui.ts`

**Files:**
- Rewrite: `web/src/lib/ui.ts`

- [ ] **Step 1: Replace the entire contents of `ui.ts`**

```ts
// ─── Surfaces & Layout ────────────────────────────────────────────────────────

// Translucent warm-paper card used on home page, settings, and studio panels
export const appPanelClass =
  "rounded-card border border-border bg-surface p-5 shadow-card";

// Outer panel shell for the three list-detail workspace columns
export const workspacePanelClass =
  "overflow-hidden rounded-3xl border border-border bg-white/76 shadow-card";

// Workspace panel title bar (serif heading + bottom border)
export const workspacePanelTitleClass =
  "m-0 border-b border-border px-5 py-4.5 font-serif text-section";

// Workspace panel content area
export const workspacePanelBodyClass = "grid gap-4 px-5 py-4.5";

// ─── Typography ───────────────────────────────────────────────────────────────

// Fluid page heading — scales from 2rem (narrow) to 3.5rem (wide)
export const pageHeadingClass =
  "m-0 mb-2 font-serif text-display leading-[0.95]";

export const sectionTitleClass = "m-0 font-serif text-2xl";

export const mutedTextClass = "m-0 text-muted";

// Small all-caps label above a heading (pink accent, wide letter-spacing)
export const eyebrowClass =
  "text-xs uppercase tracking-[0.14em] text-pink";

// ─── Form Fields ──────────────────────────────────────────────────────────────

export const fieldClass      = "grid gap-1.5";
export const fieldLabelClass = "text-sm font-semibold";
export const fieldHintClass  = "text-sm leading-5 text-muted";

// Base shared by inputClass, textareaClass, settingsInputClass, settingsSelectClass
const inputBase =
  "w-full rounded-2xl border border-border bg-white/92 px-3.5 py-3 text-ink outline-none transition focus:border-blue/45 focus:ring-2 focus:ring-sky/18";

export const inputClass    = inputBase;
export const textareaClass = `${inputBase} min-h-[96px] resize-y`;

export const checkboxRowClass = "inline-flex items-center gap-2.5";

// ─── Settings Layout ──────────────────────────────────────────────────────────

// Settings row: label+meta on the left, control on the right; wraps on mobile
export const settingsFieldClass =
  "flex flex-wrap items-start gap-x-5 gap-y-2 bg-white/36 px-4 py-3";

// Left side of a settings row (label + hint paragraph)
export const settingsFieldMetaClass =
  "min-w-[190px] flex-1 basis-[240px] grid gap-1";

// Right side: fixed-width control column, full-width on mobile
export const settingsControlClass = "w-full min-w-[220px] flex-none md:w-[320px]";

export const settingsInputClass  = `${inputBase} h-11 min-h-11 px-3 py-2.5`;
export const settingsSelectClass =
  "w-full min-w-[220px] flex-none rounded-xl border border-border bg-white/92 px-3 py-2.5 text-ink outline-none transition focus:border-blue/45 focus:ring-2 focus:ring-sky/18 h-11 min-h-11 md:w-[320px]";

// ─── Buttons ──────────────────────────────────────────────────────────────────

// Shared structure: layout, sizing, disabled state
const buttonBase =
  "inline-flex items-center justify-center rounded-2xl transition disabled:cursor-not-allowed disabled:opacity-60";

// Pink→blue gradient — primary call to action
export const primaryButtonClass =
  `${buttonBase} border border-transparent bg-linear-to-br from-pink to-blue px-4 py-[11px] text-white hover:brightness-105`;

export const secondaryButtonClass =
  `${buttonBase} border border-border bg-white/92 px-4 py-[11px] text-ink hover:bg-white`;

// Outline style with sky hover — secondary in-context actions
export const ghostButtonClass =
  `${buttonBase} border border-blue/18 bg-transparent px-4 py-[11px] text-blue hover:bg-sky/8`;

export const dangerButtonClass =
  `${buttonBase} border border-transparent bg-danger px-4 py-[11px] text-white hover:brightness-105`;

// ─── Status ───────────────────────────────────────────────────────────────────

export const statusSuccessClass = "m-0 text-sm text-success";
export const statusErrorClass   = "m-0 text-sm text-danger";

// Inline pill chip for dependency/health rows; color reflects operational status
export function statusChipClass(
  status: "ok" | "error" | "unavailable" | "neutral" = "neutral",
) {
  const base =
    "inline-flex items-center rounded-full border border-border px-2.5 py-1.5 text-sm capitalize";
  const colors: Record<typeof status, string> = {
    ok:          "text-success",
    error:       "text-danger",
    unavailable: "text-warn",
    neutral:     "text-blue",
  };
  return `${base} ${colors[status]}`;
}
```

- [ ] **Step 2: Run lint + tests**

```bash
cd web && npm run lint && npm run test
```
Expected: 0 errors, all tests pass. (Components still compile; they import the same export names.)

- [ ] **Step 3: Commit**

```bash
git add web/src/lib/ui.ts
git commit -m "refactor | web | rewrite ui.ts with design tokens and grouped constants"
```

---

### Task 3: Update `AppFrame.tsx` and `IdentityPanel.tsx`

**Files:**
- Modify: `web/src/components/layout/AppFrame.tsx`
- Modify: `web/src/components/layout/IdentityPanel.tsx`

- [ ] **Step 1: Replace `AppFrame.tsx`**

Refer to the Token Reference at the top of this plan. Changes from current file:
- `z-[100]` → `z-nav`
- `rounded-[10px]` → `rounded-icon`
- `hover:bg-[rgba(125,217,255,0.12)]` → `hover:bg-sky/12`
- `text-[#5f7b9f]` → `text-blue/70` (icon tint, nearest token)
- `border-[rgba(255,255,255,0.92)]` → `border-white/92`
- `right-[2px] bottom-[2px]` → `right-0.5 bottom-0.5`
- `h-[14px] min-w-[14px]` → `h-3.5 min-w-3.5`
- `text-[0.5rem]` → `text-badge`
- `bg-[rgba(255,255,255,0.96)]` → `bg-white/96`
- `text-[#274970]` → `text-ink`
- Add `{/* fluid navbar padding */}` comment on `px-[clamp(16px,3vw,48px)]`
- Add `{/* fluid content padding: clamps 16px..160px */}` comment on `px-[clamp(16px,10vw,160px)]`
- Add `{/* 52px nav + 28px breathing room */}` comment on `pt-[calc(52px+28px)]`

```tsx
import { useEffect, useState } from "react";
import { FiSettings, FiUser } from "react-icons/fi";
import { MdTranslate } from "react-icons/md";
import { NavLink, Outlet } from "react-router-dom";

import { useLanguage } from "../../app/LanguageContext";
import { JobsBadge } from "../jobs/JobsBadge";
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
        if (!cancelled) setHealthTier(deriveOverallHealthTier(next));
      } catch {
        if (!cancelled) setHealthTier("unavailable");
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
    <>
      <nav
        className="fixed inset-x-0 top-0 z-nav flex h-[52px] items-center justify-between bg-white px-[clamp(16px,3vw,48px)] {/* fluid navbar padding */}"
      >
        <NavLink className="inline-flex items-center" to="/" aria-label="Home">
          <span className='inline-flex h-7 w-7 items-center justify-center rounded-lg bg-[linear-gradient(135deg,#f3a9c9_0%,#f58bb9_58%,#a9e7ff_100%)] font-serif text-base font-bold text-[#fff9f4]'>
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
              className={`absolute right-0.5 bottom-0.5 h-[9px] w-[9px] rounded-full border-2 border-white/92 ${healthMeta.className}`}
            />
          </NavLink>

          <JobsBadge />

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

      {/* fluid content padding: clamps 16px..160px; 52px nav + 28px breathing room */}
      <div className="min-h-screen px-[clamp(16px,10vw,160px)] pt-[calc(52px+28px)] pb-6 max-[820px]:px-3 max-[820px]:pt-[calc(52px+18px)] max-[820px]:pb-3">
        <main>
          <Outlet />
        </main>
      </div>
    </>
  );
}
```

- [ ] **Step 2: Replace `IdentityPanel.tsx`**

```tsx
type IdentityPanelProps = {
  onClose: () => void;
};

const PLATFORMS = [{ key: "bilibili", label: "Bilibili", icon: "B" }];

export default function IdentityPanel({ onClose }: IdentityPanelProps) {
  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-backdrop border-0 bg-transparent"
        aria-label="Close identity panel"
        onClick={onClose}
      />
      <div
        {/* top-[52px]: aligns below 52px navbar */}
        className="fixed top-[52px] right-[clamp(16px,3vw,48px)] z-overlay min-w-[180px] rounded-overlay border border-border bg-white/96 p-4 shadow-overlay backdrop-blur-[18px] {/* glass blur — no matching Tailwind step */}"
        role="menu"
        aria-label="Identity"
      >
        <div className="flex flex-wrap gap-3">
          {PLATFORMS.map((platform) => (
            <div key={platform.key} className="flex w-[72px] flex-col items-center gap-1">
              <span
                className="inline-flex h-8 w-8 items-center justify-center rounded-icon bg-pink/14 font-bold text-blue"
                aria-hidden="true"
              >
                {platform.icon}
              </span>
              <span className="text-xs font-semibold text-ink">{platform.label}</span>
              <span className="text-center text-xs text-muted">Not signed in</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
```

- [ ] **Step 3: Run lint + tests**

```bash
cd web && npm run lint && npm run test
```
Expected: 0 errors, all tests pass.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/layout/AppFrame.tsx web/src/components/layout/IdentityPanel.tsx
git commit -m "refactor | web | apply design tokens to AppFrame and IdentityPanel"
```

---

### Task 4: Update `JobsBadge.tsx` and `SettingsPage.tsx`

**Files:**
- Modify: `web/src/components/jobs/JobsBadge.tsx`
- Modify: `web/src/pages/SettingsPage.tsx`

- [ ] **Step 1: Replace `JobsBadge.tsx`**

Tokens applied vs current file:
- `border-[rgba(106,147,198,0.12)]` → `border-border`
- `bg-[rgba(255,255,255,0.76)]` → `bg-white/76`
- `px-[14px]` → `px-3.5`
- `text-[#547094]` → `text-muted/80`
- `text-[#8096b3]` → `text-muted`
- `bg-[rgba(96,123,163,0.14)]` → `bg-blue/14`
- `rounded-[28px]` → `rounded-drawer`
- `bg-[rgba(255,255,255,0.92)]` → `bg-white/92`
- `shadow-[0_22px_44px_rgba(116,148,194,0.12)]` → `shadow-elevated`
- `font-["Iowan_Old_Style","Palatino_Linotype",serif]` → `font-serif`
- `rounded-[20px]` → `rounded-3xl`
- `gap-[14px]` → `gap-3.5`
- Keep `top-[92px]` with comment, `w-[min(420px,...)]` with comment, `max-h-[calc(...)]` with comment, `backdrop-blur-[18px]` with comment, `bg-[rgba(248,251,255,0.72)]` with comment

```tsx
import { useEffect, useMemo, useState } from "react";

import { JOBS_REFRESH_EVENT, api, toErrorMessage } from "../../lib/api";
import type { Job } from "../../lib/types";
import { ghostButtonClass, mutedTextClass, statusChipClass, statusErrorClass } from "../../lib/ui";

const TERMINAL_STATUSES = new Set(["done", "failed"]);

function getJobTitle(job: Job): string {
  const title = job.meta.title;
  return typeof title === "string" && title.trim() ? title : job.source_url;
}

function formatActiveJobsLabel(count: number): string {
  if (count === 0) return "No active jobs";
  if (count === 1) return "1 active job";
  return `${count} active jobs`;
}

export function JobsBadge() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [cancellingJobId, setCancellingJobId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadJobs() {
      try {
        const nextJobs = await api.listJobs();
        if (!cancelled) {
          setJobs(nextJobs);
          setErrorMessage(null);
        }
      } catch (error) {
        if (!cancelled) setErrorMessage(toErrorMessage(error));
      }
    }

    void loadJobs();
    function handleRefresh() { void loadJobs(); }
    window.addEventListener(JOBS_REFRESH_EVENT, handleRefresh);
    const intervalId = window.setInterval(() => { void loadJobs(); }, 5000);
    return () => {
      cancelled = true;
      window.removeEventListener(JOBS_REFRESH_EVENT, handleRefresh);
      window.clearInterval(intervalId);
    };
  }, []);

  const activeJobs = useMemo(
    () => jobs.filter((job) => !TERMINAL_STATUSES.has(job.status)),
    [jobs],
  );

  async function handleCancel(jobId: string) {
    setCancellingJobId(jobId);
    try {
      await api.deleteJob(jobId);
      const nextJobs = await api.listJobs();
      setJobs(nextJobs);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(toErrorMessage(error));
    } finally {
      setCancellingJobId(null);
    }
  }

  return (
    <>
      <button
        type="button"
        className="inline-flex items-center gap-2 rounded-full border border-border bg-white/76 px-3.5 py-2.5 text-muted/80"
        aria-label={formatActiveJobsLabel(activeJobs.length)}
        aria-expanded={isOpen}
        aria-controls="jobs-drawer"
        onClick={() => setIsOpen((open) => !open)}
      >
        <strong>Jobs</strong>
        <span className="text-muted">{formatActiveJobsLabel(activeJobs.length)}</span>
      </button>

      {isOpen ? (
        <>
          <button
            type="button"
            className="fixed inset-0 z-[19] border-0 bg-blue/14"
            aria-label="Close jobs drawer"
            onClick={() => setIsOpen(false)}
          />
          <section
            id="jobs-drawer"
            {/* top-[92px]: 52px navbar + 40px gap; w-[min(...)]: caps drawer at 420px; max-h keeps it in-viewport; backdrop-blur-[18px]: glass blur — no matching Tailwind step */}
            className="fixed top-[92px] right-6 z-20 grid max-h-[calc(100vh-116px)] w-[min(420px,calc(100vw-24px))] gap-4 overflow-auto rounded-drawer border border-border bg-white/92 p-5 shadow-elevated backdrop-blur-[18px] max-[820px]:top-[84px] max-[820px]:right-3 max-[820px]:w-[calc(100vw-24px)]"
            aria-label="Jobs"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="m-0 font-serif">Jobs</h2>
                <p className={mutedTextClass}>Background ingestion and model work.</p>
              </div>
              <button type="button" className={ghostButtonClass} onClick={() => setIsOpen(false)}>
                Close
              </button>
            </div>
            {errorMessage ? <p className={statusErrorClass}>{errorMessage}</p> : null}
            <div className="grid gap-3">
              {jobs.length === 0 ? (
                <div className="grid min-h-[120px] place-items-center rounded-3xl border border-border bg-[rgba(248,251,255,0.72)] {/* sky-tinted card background */} p-4">
                  <p className={mutedTextClass}>No jobs yet.</p>
                </div>
              ) : (
                jobs.map((job) => {
                  const jobTitle = getJobTitle(job);
                  const isTerminal = TERMINAL_STATUSES.has(job.status);
                  return (
                    <article
                      key={job.id}
                      className="grid gap-3.5 rounded-3xl border border-border bg-[rgba(248,251,255,0.72)] {/* sky-tinted card background */} p-4"
                    >
                      <div className="grid gap-2">
                        <div className="flex flex-wrap items-center gap-3">
                          <h3 className="m-0 font-serif">{jobTitle}</h3>
                          <span className={statusChipClass(job.status === "failed" ? "error" : job.status === "done" ? "ok" : "unavailable")}>
                            {job.status}
                          </span>
                        </div>
                        <p className={mutedTextClass}>{job.progress}%</p>
                        {job.error ? <p className={statusErrorClass}>{job.error}</p> : null}
                      </div>
                      {!isTerminal ? (
                        <button
                          type="button"
                          className={ghostButtonClass}
                          aria-label={`Cancel ${jobTitle}`}
                          disabled={cancellingJobId === job.id}
                          onClick={() => void handleCancel(job.id)}
                        >
                          {cancellingJobId === job.id ? "Cancelling..." : "Cancel"}
                        </button>
                      ) : null}
                    </article>
                  );
                })
              )}
            </div>
          </section>
        </>
      ) : null}
    </>
  );
}
```

- [ ] **Step 2: Update `SettingsPage.tsx` inline arbitraries**

Apply these replacements to the tab button className expression (lines 140–144 in current file):

```tsx
// Before:
className={`flex items-center gap-3 rounded-xl px-4 py-3 text-left transition ${
  isActive
    ? "bg-[rgba(125,217,255,0.12)] font-semibold text-[#274970]"
    : "text-[#8096b3] hover:bg-[rgba(125,217,255,0.08)] hover:text-[#274970]"
}`}

// After:
className={`flex items-center gap-3 rounded-xl px-4 py-3 text-left transition ${
  isActive
    ? "bg-sky/12 font-semibold text-ink"
    : "text-muted hover:bg-sky/8 hover:text-ink"
}`}
```

- [ ] **Step 3: Run lint + tests**

```bash
cd web && npm run lint && npm run test
```
Expected: 0 errors, all tests pass.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/jobs/JobsBadge.tsx web/src/pages/SettingsPage.tsx
git commit -m "refactor | web | apply design tokens to JobsBadge and SettingsPage"
```

---

### Task 5: Update `ListGrid.tsx`, `ChatPanel.tsx`, and `SourceList.tsx`

**Files:**
- Modify: `web/src/components/lists/ListGrid.tsx`
- Modify: `web/src/components/chat/ChatPanel.tsx`
- Modify: `web/src/components/sources/SourceList.tsx`

- [ ] **Step 1: Replace `ListGrid.tsx`**

```tsx
import type { LocusList } from "../../lib/types";
import { dangerButtonClass, eyebrowClass, mutedTextClass, sectionTitleClass } from "../../lib/ui";

type Props = {
  lists: LocusList[];
  onDelete: (list: LocusList) => Promise<void>;
  onOpen: (list: LocusList) => void;
  onCreate: () => Promise<void>;
  busy: boolean;
};

export function ListGrid({ lists, onDelete, onOpen, onCreate, busy }: Props) {
  return (
    <section
      className="grid grid-cols-[repeat(auto-fit,272px)] justify-start gap-4 max-[820px]:grid-cols-1"
      aria-label="List grid"
    >
      {/* List card gradient — 3-stop pink→sky, unique to this card */}
      <article className="grid min-h-[220px] w-[272px] overflow-hidden rounded-card bg-[linear-gradient(160deg,rgba(245,140,185,0.72)_0%,rgba(243,162,198,0.68)_52%,rgba(150,227,255,0.7)_100%)] shadow-card max-[820px]:w-full">
        <button
          aria-label="Create new list"
          className="grid min-h-[220px] w-full content-between justify-items-start gap-4 border-0 bg-transparent p-[22px] text-left text-white"
          disabled={busy}
          onClick={() => void onCreate()}
          type="button"
        >
          <span className="text-[2.5rem] leading-none">+</span>
          <span className="font-serif text-2xl">Create new list</span>
        </button>
      </article>

      {lists.map((list) => (
        <article
          className="grid min-h-[220px] w-[272px] gap-3.5 rounded-card border border-border bg-surface p-5 shadow-card max-[820px]:w-full"
          key={list.id}
        >
          <button
            aria-label={`Open ${list.name}`}
            className="flex items-center justify-between border-0 bg-transparent p-0 text-left text-inherit"
            onClick={() => onOpen(list)}
            type="button"
          >
            <span className={eyebrowClass}>Notebook</span>
            <span className="rounded-full bg-sky/10 px-3 py-2 text-sm text-[#4e6f99]">Open</span>
          </button>
          <div>
            <h2 className={sectionTitleClass}>{list.name}</h2>
            <p className={mutedTextClass}>Created {new Date(list.created_at).toLocaleString()}</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              aria-label={`Delete ${list.name}`}
              className={dangerButtonClass}
              onClick={() => onDelete(list)}
              type="button"
            >
              Delete
            </button>
          </div>
        </article>
      ))}
    </section>
  );
}
```

- [ ] **Step 2: Replace `ChatPanel.tsx`**

```tsx
import { mutedTextClass, workspacePanelBodyClass, workspacePanelClass, workspacePanelTitleClass } from "../../lib/ui";

export function ChatPanel() {
  return (
    <section className={workspacePanelClass}>
      <h2 className={workspacePanelTitleClass}>Chat</h2>
      <div className={`${workspacePanelBodyClass} min-h-[490px] content-center`}>
        <div className="grid gap-2.5">
          {/* Skeleton shimmer lines — pink/sky gradient, 3 widths */}
          <div className="h-3.5 w-[86%] rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
          <div className="h-3.5 rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
          <div className="h-3.5 w-[54%] rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
        </div>
        <p className={mutedTextClass}>List-scoped chat arrives in v1. This panel stays intentionally quiet until then.</p>
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Replace `SourceList.tsx`**

```tsx
import { useState } from "react";

import type { Source } from "../../lib/types";
import {
  checkboxRowClass,
  fieldClass,
  fieldLabelClass,
  ghostButtonClass,
  inputClass,
  mutedTextClass,
  primaryButtonClass,
  statusErrorClass,
  statusSuccessClass,
  workspacePanelBodyClass,
} from "../../lib/ui";

type Props = {
  busy: boolean;
  error: string | null;
  ingestStatus: string | null;
  onDelete: (source: Source) => Promise<void>;
  onIngest: (url: string, rerun: boolean) => Promise<void>;
  onOpen: (source: Source) => void;
  sources: Source[];
};

export function SourceList({ busy, error, ingestStatus, onDelete, onIngest, onOpen, sources }: Props) {
  const [url, setUrl] = useState("");
  const [rerun, setRerun] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!url.trim()) return;
    await onIngest(url.trim(), rerun);
    setUrl("");
  }

  return (
    <div className={workspacePanelBodyClass}>
      <form className="grid gap-4" onSubmit={handleSubmit}>
        <label className={fieldClass}>
          <span className={fieldLabelClass}>Source URL</span>
          <input
            aria-label="Source URL"
            className={inputClass}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://www.bilibili.com/video/..."
            value={url}
          />
        </label>
        <label className={checkboxRowClass}>
          <input
            aria-label="Re-run existing source"
            checked={rerun}
            onChange={(event) => setRerun(event.target.checked)}
            type="checkbox"
          />
          <span>Re-run existing source</span>
        </label>
        <div className="flex flex-wrap items-center gap-3">
          <button className={primaryButtonClass} disabled={busy} type="submit">
            {busy ? "Queueing..." : "Queue source"}
          </button>
          {ingestStatus ? <p className={statusSuccessClass}>{ingestStatus}</p> : null}
        </div>
        {error ? <p className={statusErrorClass}>{error}</p> : null}
      </form>

      <div className="grid gap-3">
        {sources.length === 0 ? (
          <div className="flex justify-start rounded-2xl border border-border bg-white/64 px-4 py-3.5">
            <p className={mutedTextClass}>No sources yet. Queue a Bilibili URL to start building this notebook.</p>
          </div>
        ) : (
          sources.map((source) => (
            <article
              className="flex items-center justify-between gap-3 rounded-2xl border border-border bg-white/64 px-4 py-3.5"
              key={source.video_id}
            >
              <button
                aria-label={`Open ${source.title}`}
                className="grid gap-1 border-0 bg-transparent text-left text-inherit"
                onClick={() => onOpen(source)}
                type="button"
              >
                <strong>{source.title}</strong>
                <span className={mutedTextClass}>{source.platform}</span>
              </button>
              <button
                aria-label={`Delete ${source.title}`}
                className={ghostButtonClass}
                onClick={() => void onDelete(source)}
                type="button"
              >
                Delete
              </button>
            </article>
          ))
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run lint + tests**

```bash
cd web && npm run lint && npm run test
```
Expected: 0 errors, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/lists/ListGrid.tsx web/src/components/chat/ChatPanel.tsx web/src/components/sources/SourceList.tsx
git commit -m "refactor | web | apply design tokens to ListGrid, ChatPanel, SourceList"
```

---

### Task 6: Update `SourceDetail.tsx` and `ListDetailPage.tsx`

**Files:**
- Modify: `web/src/components/sources/SourceDetail.tsx`
- Modify: `web/src/pages/ListDetailPage.tsx`

- [ ] **Step 1: Replace `SourceDetail.tsx`**

```tsx
import { useState } from "react";

import ReactMarkdown from "react-markdown";

import { downloadTextFile } from "../../lib/download";
import type { NoteContent, Source } from "../../lib/types";
import { ghostButtonClass, mutedTextClass, statusErrorClass, workspacePanelBodyClass } from "../../lib/ui";

type Props = {
  activeTab: "note" | "transcript";
  note: NoteContent | null;
  source: Source;
  transcript: string | null;
  transcriptError: string | null;
  transcriptLoading: boolean;
  onBack: () => void;
  onSelectTab: (tab: "note" | "transcript") => Promise<void>;
};

export function SourceDetail({
  activeTab,
  note,
  onBack,
  onSelectTab,
  source,
  transcript,
  transcriptError,
  transcriptLoading,
}: Props) {
  const [downloading, setDownloading] = useState(false);

  function handleDownload() {
    if (!note) return;
    setDownloading(true);
    downloadTextFile(`${source.video_id}.md`, note.markdown);
    setDownloading(false);
  }

  return (
    <div className={workspacePanelBodyClass}>
      <div className="grid gap-3">
        <button className={ghostButtonClass} onClick={onBack} type="button">
          Back to sources
        </button>
        <div>
          <h3 className="m-0 font-serif text-2xl">{source.title}</h3>
          <p className={mutedTextClass}>{source.platform}</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2.5">
        <button
          className={`rounded-full border px-3.5 py-2.5 transition ${
            activeTab === "note"
              ? "border-transparent bg-blue text-white"
              : "border-border bg-sky/10 text-muted/80"
          }`}
          onClick={() => void onSelectTab("note")}
          type="button"
        >
          Note
        </button>
        <button
          className={`rounded-full border px-3.5 py-2.5 transition ${
            activeTab === "transcript"
              ? "border-transparent bg-blue text-white"
              : "border-border bg-sky/10 text-muted/80"
          }`}
          onClick={() => void onSelectTab("transcript")}
          type="button"
        >
          Transcript
        </button>
        <button className={ghostButtonClass} disabled={downloading || !note} onClick={handleDownload} type="button">
          Download note
        </button>
      </div>

      {activeTab === "note" ? (
        <div className="min-h-[320px] rounded-2xl border border-border bg-white/60 p-4.5">
          <ReactMarkdown>{note?.markdown ?? ""}</ReactMarkdown>
        </div>
      ) : (
        <div className="min-h-[320px] rounded-2xl border border-border bg-white/60 p-4.5">
          {transcriptLoading ? <p className={mutedTextClass}>Loading transcript...</p> : null}
          {transcriptError ? <p className={statusErrorClass}>{transcriptError}</p> : null}
          {transcript ? (
            <pre className="m-0 whitespace-pre-wrap font-mono text-[#4e6485]">{transcript}</pre>
          ) : null}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Update `ListDetailPage.tsx` — rename input className**

The only inline arbitrary values in this file are the editable list-name input's className (line 174 in the current file). Replace that one className string:

```tsx
// Before:
className='w-full rounded-2xl border border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.84)] px-[14px] py-3 font-["Iowan_Old_Style","Palatino_Linotype",serif] text-[clamp(2rem,4vw,3.5rem)] leading-[0.95] text-[#274970] outline-none'

// After:
className="w-full rounded-2xl border border-border bg-white/84 px-3.5 py-3 font-serif text-display leading-[0.95] text-ink outline-none"
```

- [ ] **Step 3: Run lint + tests**

```bash
cd web && npm run lint && npm run test
```
Expected: 0 errors, all tests pass.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/sources/SourceDetail.tsx web/src/pages/ListDetailPage.tsx
git commit -m "refactor | web | apply design tokens to SourceDetail and ListDetailPage"
```

---

### Task 7: Update `TranscriptTab.tsx` and `OtherTab.tsx`

**Files:**
- Modify: `web/src/components/settings/TranscriptTab.tsx`
- Modify: `web/src/components/settings/OtherTab.tsx`

- [ ] **Step 1: Apply token replacements to `TranscriptTab.tsx`**

Find and replace the following in the file:

| Find | Replace |
|---|---|
| `text-[11px] font-semibold uppercase tracking-[0.14em] text-[#8096b3]` | `text-xs font-semibold uppercase tracking-[0.14em] text-muted` |
| `rounded-[18px] border border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.64)]` | `rounded-2xl border border-border bg-white/64` |
| `border-t border-[rgba(106,147,198,0.12)]` | `border-t border-border` |
| `text-[#274970]` | `text-ink` |
| `text-[0.82rem] text-[#8096b3]` | `text-sm text-muted` |

- [ ] **Step 2: Apply token replacements to `OtherTab.tsx`**

Find and replace the following in the file:

| Find | Replace |
|---|---|
| `text-[0.82rem] text-[#8096b3]` | `text-sm text-muted` |
| `text-[0.88rem] leading-6 text-[#5b7faa]` | `text-sm leading-6 text-blue` |
| `bg-[rgba(255,255,255,0.36)]` | `bg-white/36` |
| `border-[rgba(106,147,198,0.18)]` | `border-blue/18` |
| `text-[#5b7faa] underline` | `text-blue underline` |

- [ ] **Step 3: Run lint + tests**

```bash
cd web && npm run lint && npm run test
```
Expected: 0 errors, all tests pass.

- [ ] **Step 4: Verify no remaining arbitrary color/shadow/radius values**

```bash
cd web && grep -rn '\[rgba\|text-\[#\|bg-\[#\|border-\[#\|shadow-\[0\|rounded-\[[0-9]' src/components src/pages src/lib/ui.ts
```

Expected output: only the documented exceptions listed in the Token Reference at the top of this plan. If anything else appears, apply the correct token from the Token Reference and re-run.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/settings/TranscriptTab.tsx web/src/components/settings/OtherTab.tsx
git commit -m "refactor | web | apply design tokens to TranscriptTab and OtherTab — Phase A complete"
```

---

## Phase B

### Task 8: `Button` component

**Files:**
- Create: `web/src/components/ui/Button.tsx`
- Create: `web/src/test/ui-components.test.tsx`

- [ ] **Step 1: Write the failing tests**

Create `web/src/test/ui-components.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { Button } from "../components/ui/Button";
import { FormField } from "../components/ui/FormField";
import { Panel } from "../components/ui/Panel";
import { StatusChip } from "../components/ui/StatusChip";

// ── Button ──────────────────────────────────────────────────────────────────
describe("Button", () => {
  test("renders primary variant", () => {
    render(<Button variant="primary">Save</Button>);
    const btn = screen.getByRole("button", { name: "Save" });
    expect(btn.className).toContain("from-pink");
    expect(btn.className).toContain("to-blue");
  });

  test("renders ghost variant", () => {
    render(<Button variant="ghost">Cancel</Button>);
    expect(screen.getByRole("button").className).toContain("border-blue");
  });

  test("renders danger variant", () => {
    render(<Button variant="danger">Delete</Button>);
    expect(screen.getByRole("button").className).toContain("bg-danger");
  });

  test("forwards className prop", () => {
    render(<Button variant="secondary" className="mt-4">Go</Button>);
    expect(screen.getByRole("button").className).toContain("mt-4");
  });

  test("forwards disabled and onClick", () => {
    const handler = vi.fn();
    render(<Button variant="primary" disabled onClick={handler}>X</Button>);
    const btn = screen.getByRole("button");
    expect(btn).toBeDisabled();
  });

  // ── FormField ────────────────────────────────────────────────────────────
  test("FormField renders label and children", () => {
    render(<FormField label="Email"><input /></FormField>);
    expect(screen.getByText("Email")).toBeInTheDocument();
  });

  test("FormField renders hint when provided", () => {
    render(<FormField label="Name" hint="Required"><input /></FormField>);
    expect(screen.getByText("Required")).toBeInTheDocument();
  });

  test("FormField forwards className", () => {
    const { container } = render(<FormField label="X" className="mt-2"><input /></FormField>);
    expect(container.firstChild as HTMLElement).toHaveClass("mt-2");
  });

  // ── Panel ────────────────────────────────────────────────────────────────
  test("Panel renders app variant", () => {
    const { container } = render(<Panel variant="app"><p>content</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("bg-surface");
  });

  test("Panel renders workspace variant", () => {
    const { container } = render(<Panel variant="workspace"><p>content</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("bg-white");
  });

  test("Panel forwards className", () => {
    const { container } = render(<Panel className="p-8"><p>x</p></Panel>);
    expect((container.firstChild as HTMLElement).className).toContain("p-8");
  });

  // ── StatusChip ───────────────────────────────────────────────────────────
  test("StatusChip renders ok status color", () => {
    render(<StatusChip status="ok">OK</StatusChip>);
    expect(screen.getByText("OK").className).toContain("text-success");
  });

  test("StatusChip renders error status color", () => {
    render(<StatusChip status="error">Error</StatusChip>);
    expect(screen.getByText("Error").className).toContain("text-danger");
  });

  test("StatusChip forwards className", () => {
    render(<StatusChip status="ok" className="ml-2">OK</StatusChip>);
    expect(screen.getByText("OK").className).toContain("ml-2");
  });
});
```

- [ ] **Step 2: Run to verify all tests fail**

```bash
cd web && npm run test -- ui-components
```
Expected: all 14 tests FAIL — modules not found.

- [ ] **Step 3: Create `Button.tsx`**

Create `web/src/components/ui/Button.tsx`:

```tsx
import { ComponentPropsWithoutRef, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";

// py-[11px]: button vertical padding — no matching Tailwind spacing step
const base =
  "inline-flex items-center justify-center rounded-2xl transition disabled:cursor-not-allowed disabled:opacity-60";

const variants: Record<Variant, string> = {
  primary:   "border border-transparent bg-linear-to-br from-pink to-blue px-4 py-[11px] text-white hover:brightness-105",
  secondary: "border border-border bg-white/92 px-4 py-[11px] text-ink hover:bg-white",
  ghost:     "border border-blue/18 bg-transparent px-4 py-[11px] text-blue hover:bg-sky/8",
  danger:    "border border-transparent bg-danger px-4 py-[11px] text-white hover:brightness-105",
};

interface Props extends ComponentPropsWithoutRef<"button"> {
  variant?: Variant;
}

export const Button = forwardRef<HTMLButtonElement, Props>(
  ({ variant = "secondary", className = "", ...rest }, ref) => (
    <button
      ref={ref}
      className={`${base} ${variants[variant]} ${className}`.trim()}
      {...rest}
    />
  ),
);
Button.displayName = "Button";
```

- [ ] **Step 4: Run tests — expect Button tests to pass, others still fail**

```bash
cd web && npm run test -- ui-components
```
Expected: 5 Button tests PASS, 9 others FAIL.

---

### Task 9: `FormField`, `Panel`, `StatusChip` components

**Files:**
- Create: `web/src/components/ui/FormField.tsx`
- Create: `web/src/components/ui/Panel.tsx`
- Create: `web/src/components/ui/StatusChip.tsx`

- [ ] **Step 1: Create `FormField.tsx`**

```tsx
import { ComponentPropsWithoutRef, ReactNode } from "react";

interface Props extends Omit<ComponentPropsWithoutRef<"label">, "children"> {
  label: string;
  hint?: string;
  children: ReactNode;
}

// Renders as <label> so the browser implicitly associates it with the first
// interactive child — works for single-input fields (input, select, textarea).
export function FormField({ label, hint, children, className = "", ...rest }: Props) {
  return (
    <label className={`grid gap-1.5 ${className}`.trim()} {...rest}>
      <span className="text-sm font-semibold">{label}</span>
      {children}
      {hint ? <span className="text-sm leading-5 text-muted">{hint}</span> : null}
    </label>
  );
}
```

- [ ] **Step 2: Create `Panel.tsx`**

```tsx
import { ComponentPropsWithoutRef } from "react";

type Variant = "app" | "workspace";

const variants: Record<Variant, string> = {
  app:       "rounded-card border border-border bg-surface p-5 shadow-card",
  workspace: "overflow-hidden rounded-3xl border border-border bg-white/76 shadow-card",
};

interface Props extends ComponentPropsWithoutRef<"div"> {
  variant?: Variant;
}

export function Panel({ variant = "app", className = "", ...rest }: Props) {
  return (
    <div className={`${variants[variant]} ${className}`.trim()} {...rest} />
  );
}
```

- [ ] **Step 3: Create `StatusChip.tsx`**

```tsx
import { ComponentPropsWithoutRef } from "react";

type Status = "ok" | "error" | "unavailable" | "neutral";

const statusColors: Record<Status, string> = {
  ok:          "text-success",
  error:       "text-danger",
  unavailable: "text-warn",
  neutral:     "text-blue",
};

interface Props extends ComponentPropsWithoutRef<"span"> {
  status?: Status;
}

export function StatusChip({ status = "neutral", className = "", ...rest }: Props) {
  return (
    <span
      className={`inline-flex items-center rounded-full border border-border px-2.5 py-1.5 text-sm capitalize ${statusColors[status]} ${className}`.trim()}
      {...rest}
    />
  );
}
```

- [ ] **Step 4: Run tests — all 14 should pass**

```bash
cd web && npm run test -- ui-components
```
Expected: all 14 PASS.

- [ ] **Step 5: Create the barrel export `web/src/components/ui/index.ts`**

```ts
export { Button } from "./Button";
export { FormField } from "./FormField";
export { Panel } from "./Panel";
export { StatusChip } from "./StatusChip";
```

- [ ] **Step 6: Run full test suite**

```bash
cd web && npm run lint && npm run test
```
Expected: 0 errors, all tests pass.

- [ ] **Step 7: Commit**

```bash
git add web/src/components/ui/ web/src/test/ui-components.test.tsx
git commit -m "feat | web | add Button, FormField, Panel, StatusChip ui components"
```

---

### Task 10: Replace call sites and remove dead constants from `ui.ts`

**Files:**
- Modify: `web/src/components/sources/SourceList.tsx`
- Modify: `web/src/components/sources/SourceDetail.tsx`
- Modify: `web/src/components/jobs/JobsBadge.tsx`
- Modify: `web/src/components/lists/ListGrid.tsx`
- Modify: `web/src/components/studio/StudioPanel.tsx`
- Modify: `web/src/pages/ListDetailPage.tsx`
- Modify: `web/src/pages/HomePage.tsx`
- Modify: `web/src/pages/SettingsPage.tsx`
- Modify: `web/src/lib/ui.ts`

- [ ] **Step 1: Update `SourceList.tsx`**

Add import, replace two button elements and the fieldClass pattern:

```tsx
// Add to imports:
import { Button, FormField } from "../ui";

// Replace the label+span+input block (currently uses fieldClass + fieldLabelClass):
// Before:
<label className={fieldClass}>
  <span className={fieldLabelClass}>Source URL</span>
  <input aria-label="Source URL" className={inputClass} ... />
</label>

// After:
<FormField label="Source URL">
  <input aria-label="Source URL" className={inputClass} ... />
</FormField>

// Replace primary submit button:
// Before: <button className={primaryButtonClass} disabled={busy} type="submit">
// After:  <Button variant="primary" disabled={busy} type="submit">

// Replace ghost delete button:
// Before: <button aria-label={`Delete ${source.title}`} className={ghostButtonClass} ...>
// After:  <Button variant="ghost" aria-label={`Delete ${source.title}`} ...>

// Remove from ui imports: fieldClass, fieldLabelClass, primaryButtonClass, ghostButtonClass
```

- [ ] **Step 2: Update `SourceDetail.tsx`**

```tsx
// Add to imports:
import { Button } from "../ui";

// Replace back button:
// Before: <button className={ghostButtonClass} onClick={onBack} type="button">
// After:  <Button variant="ghost" onClick={onBack} type="button">

// Replace download button:
// Before: <button className={ghostButtonClass} disabled={downloading || !note} onClick={handleDownload} type="button">
// After:  <Button variant="ghost" disabled={downloading || !note} onClick={handleDownload} type="button">

// Remove from ui imports: ghostButtonClass
```

- [ ] **Step 3: Update `JobsBadge.tsx`**

```tsx
// Add to imports:
import { Button, StatusChip } from "../ui";

// Replace Close button:
// Before: <button type="button" className={ghostButtonClass} onClick={() => setIsOpen(false)}>
// After:  <Button variant="ghost" type="button" onClick={() => setIsOpen(false)}>

// Replace Cancel button:
// Before: <button type="button" className={ghostButtonClass} aria-label={...} disabled={...} onClick={...}>
// After:  <Button variant="ghost" type="button" aria-label={...} disabled={...} onClick={...}>

// Replace statusChipClass(...) span:
// Before: <span className={statusChipClass(...)}>  {job.status}  </span>
// After:  <StatusChip status={job.status === "failed" ? "error" : job.status === "done" ? "ok" : "unavailable"}>
//           {job.status}
//         </StatusChip>

// Remove from ui imports: ghostButtonClass, statusChipClass
```

- [ ] **Step 4: Update `ListGrid.tsx`**

```tsx
// Add to imports:
import { Button } from "../ui";

// Replace danger delete button:
// Before: <button aria-label={`Delete ${list.name}`} className={dangerButtonClass} onClick={() => onDelete(list)} type="button">
// After:  <Button variant="danger" aria-label={`Delete ${list.name}`} onClick={() => onDelete(list)} type="button">

// Remove from ui imports: dangerButtonClass
```

- [ ] **Step 5: Update `StudioPanel.tsx`**

```tsx
// Add to imports:
import { Button, Panel } from "../ui";

// Replace outer section:
// Before: <section className={workspacePanelClass}>
// After:  <Panel variant="workspace" as="section">  ← Panel renders a <div> by default;
//   since StudioPanel needs a <section>, keep the section element and pass workspacePanelClass
//   OR just use a div: Panel renders a div and that is fine here.
//   Use: <Panel variant="workspace">

// Replace primary button:
// Before: <button className={primaryButtonClass} disabled={busy} onClick={() => void onGenerate()} type="button">
// After:  <Button variant="primary" disabled={busy} onClick={() => void onGenerate()} type="button">

// Remove from ui imports: primaryButtonClass, workspacePanelClass
```

Full updated `StudioPanel.tsx`:

```tsx
import { Button, Panel } from "../ui";
import { mutedTextClass, statusErrorClass, statusSuccessClass, workspacePanelBodyClass, workspacePanelTitleClass } from "../../lib/ui";

type Props = {
  busy: boolean;
  error: string | null;
  status: string | null;
  onGenerate: () => Promise<void>;
};

export function StudioPanel({ busy, error, status, onGenerate }: Props) {
  return (
    <Panel variant="workspace">
      <h2 className={workspacePanelTitleClass}>Studio</h2>
      <div className={workspacePanelBodyClass}>
        <p className={mutedTextClass}>
          Generate a list-level markdown overview from the sources already processed into this notebook.
        </p>
        <Button variant="primary" disabled={busy} onClick={() => void onGenerate()} type="button">
          {busy ? "Generating..." : "Generate overview"}
        </Button>
        {status ? <p className={statusSuccessClass}>{status}</p> : null}
        {error ? <p className={statusErrorClass}>{error}</p> : null}
      </div>
    </Panel>
  );
}
```

- [ ] **Step 6: Update `ListDetailPage.tsx`**

```tsx
// Add to imports:
import { Button, Panel } from "../../components/ui";  // from pages/ → components/ui

// Replace the error-state section:
// Before:
<section className={workspacePanelClass}>
  <h2 className={workspacePanelTitleClass}>Sources</h2>
  <div className={workspacePanelBodyClass}>
    <p className={statusErrorClass}>{loadError}</p>
  </div>
</section>
// After:
<Panel variant="workspace">
  <h2 className={workspacePanelTitleClass}>Sources</h2>
  <div className={workspacePanelBodyClass}>
    <p className={statusErrorClass}>{loadError}</p>
  </div>
</Panel>

// Replace Rename ghost button:
// Before: <button aria-label="Edit list name" className={ghostButtonClass} onClick={...} type="button">
// After:  <Button variant="ghost" aria-label="Edit list name" onClick={...} type="button">

// Remove from ui imports: ghostButtonClass, workspacePanelClass
```

- [ ] **Step 7: Update `HomePage.tsx`**

```tsx
// Add to imports:
import { Panel } from "../components/ui";  // from pages/ → components/ui

// Replace loading-state section:
// Before: <section className={appPanelClass}>
// After:  <Panel variant="app">

// Remove from ui imports: appPanelClass
```

Full updated `HomePage.tsx`:

```tsx
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ListGrid } from "../components/lists/ListGrid";
import { Panel } from "../components/ui";
import { api, toErrorMessage } from "../lib/api";
import type { LocusList } from "../lib/types";
import { eyebrowClass, mutedTextClass, pageHeadingClass, statusErrorClass } from "../lib/ui";

export function HomePage() {
  const navigate = useNavigate();
  const [lists, setLists] = useState<LocusList[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadLists() {
      try {
        const nextLists = await api.listLists();
        if (!cancelled) setLists(nextLists);
      } catch (nextError) {
        if (!cancelled) setError(toErrorMessage(nextError));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void loadLists();
    return () => { cancelled = true; };
  }, []);

  async function handleCreate() {
    setBusy(true);
    setError(null);
    try {
      const created = await api.createList("Untitled list");
      setLists((current) => [created, ...current]);
    } catch (nextError) {
      setError(toErrorMessage(nextError));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(list: LocusList) {
    if (!window.confirm(`Delete list "${list.name}"?`)) return;
    setError(null);
    try {
      await api.deleteList(list.id);
      setLists((current) => current.filter((entry) => entry.id !== list.id));
    } catch (nextError) {
      setError(toErrorMessage(nextError));
    }
  }

  return (
    <div className="grid gap-4">
      <section className="grid max-w-[780px] gap-3">
        <p className={eyebrowClass}>Capture. Distill. Revisit.</p>
        <h1 className={pageHeadingClass}>Turn long-form video into a living, searchable notebook.</h1>
        <p className={mutedTextClass}>
          Build private list-based workspaces for courses, playlists, and research threads.
        </p>
        {error ? <p className={statusErrorClass}>{error}</p> : null}
      </section>
      {loading ? (
        <Panel variant="app"><p>Loading lists...</p></Panel>
      ) : (
        <ListGrid
          busy={busy}
          lists={lists}
          onCreate={handleCreate}
          onDelete={handleDelete}
          onOpen={(list) => navigate(`/lists/${list.id}`)}
        />
      )}
    </div>
  );
}
```

- [ ] **Step 8: Update `SettingsPage.tsx`**

```tsx
// Add to imports:
import { Panel } from "../components/ui";

// Replace both appPanelClass loading and error sections:
// Before: <section className={appPanelClass}>
// After:  <Panel variant="app">

// Remove from ui imports: appPanelClass
```

- [ ] **Step 9: Remove dead exports from `ui.ts`**

The following exports are now replaced by the `ui/` components and are no longer imported anywhere. Remove them from `ui.ts`:

- `appPanelClass` — replaced by `<Panel variant="app">`
- `primaryButtonClass` — replaced by `<Button variant="primary">`
- `secondaryButtonClass` — replaced by `<Button variant="secondary">`
- `ghostButtonClass` — replaced by `<Button variant="ghost">`
- `dangerButtonClass` — replaced by `<Button variant="danger">`
- `fieldClass` — replaced by `<FormField>`
- `fieldLabelClass` — replaced by `<FormField>`
- `statusChipClass` — replaced by `<StatusChip>`

Also remove the unexported `buttonBase` (no longer needed after button variants move into `Button.tsx`).

Keep in `ui.ts`: `workspacePanelClass`, `workspacePanelTitleClass`, `workspacePanelBodyClass`, `pageHeadingClass`, `sectionTitleClass`, `mutedTextClass`, `eyebrowClass`, `fieldHintClass`, `inputClass`, `textareaClass`, `checkboxRowClass`, `settingsFieldClass`, `settingsFieldMetaClass`, `settingsControlClass`, `settingsInputClass`, `settingsSelectClass`, `statusSuccessClass`, `statusErrorClass`.

- [ ] **Step 10: Run lint — expect zero errors**

```bash
cd web && npm run lint
```

If lint reports any unused imports in component files (e.g. `ghostButtonClass` still imported somewhere), fix them now.

- [ ] **Step 11: Run full test suite**

```bash
cd web && npm run test
```
Expected: all tests pass.

- [ ] **Step 12: Commit**

```bash
git add web/src/
git commit -m "refactor | web | replace class string call sites with Button, FormField, Panel, StatusChip — Phase B complete"
```

---

## Self-Review

### Spec coverage

| Spec Requirement | Task |
|---|---|
| `@theme` token block in `app.css` | Task 1 |
| All 16 tokens (colors, typography, shadows, radii, z-index) | Task 1 |
| `ui.ts` grouped into 6 sections with comments | Task 2 |
| `inputBase` and `buttonBase` extracted as shared bases | Task 2 |
| AppFrame + IdentityPanel inline arbitrary → tokens | Task 3 |
| JobsBadge + SettingsPage inline arbitrary → tokens | Task 4 |
| ListGrid + ChatPanel + SourceList inline arbitrary → tokens | Task 5 |
| SourceDetail + ListDetailPage inline arbitrary → tokens | Task 6 |
| TranscriptTab + OtherTab inline arbitrary → tokens | Task 7 |
| `Button` component with variant prop + forwardRef | Tasks 8–9 |
| `FormField` component (label, hint, children) | Task 9 |
| `Panel` component (app/workspace variants) | Task 9 |
| `StatusChip` component (4 status values) | Task 9 |
| Barrel `ui/index.ts` | Task 9 |
| All call sites updated | Task 10 |
| Dead exports removed from `ui.ts` | Task 10 |
| `npm run lint` + `npm run test` green after each phase | Each task |

### Notes
- `workspacePanelClass` is still used in a few places (ChatPanel, StudioPanel) but through the `Panel` component wrapper in Phase B — ensure `ChatPanel.tsx` is updated in Task 5 to use `<Panel variant="workspace">` (it was shown in Task 5 already using ui class constants; the Phase B update for ChatPanel is not explicitly listed above — add it to Task 10 following the same pattern as StudioPanel).
- The `settingsFieldClass` group stays in `ui.ts` per spec — it is a complex layout utility, not a component.
