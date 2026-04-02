# Tailwind Design System Refactor

**Goal:** Replace all arbitrary Tailwind values with a named token layer, normalize spacing to Tailwind's built-in scale, and restructure `ui.ts` into grouped, documented constants — maximizing reuse and eliminating magic values across the frontend.

**Scope:** Two sequential phases. This spec covers both. Each phase produces a clean, reviewable diff on its own.

---

## Background

The current frontend uses Tailwind v4 but without any `@theme` token definitions. Every color, shadow, radius, and font reference is an inline arbitrary value — `text-[#8096b3]`, `border-[rgba(106,147,198,0.12)]`, `shadow-[0_14px_28px_rgba(116,148,194,0.07)]` — repeated across 14+ files. `ui.ts` centralizes some of this, but is itself entirely made of arbitrary values. There are no shared constants for colors, no spacing conventions, and no documentation on intent.

---

## Phase A: Token Layer

### Mechanism

Tailwind v4 registers tokens via an `@theme` block in `app.css`. Each CSS custom property becomes a utility class automatically:

```css
@theme {
  --color-pink: #f08bb9;  /* → text-pink, bg-pink, border-pink */
}
```

No config file needed. No plugin. Tokens are co-located with the base layer in `app.css`.

### Token Definitions

**Colors**

| Token | Value | Replaces |
|---|---|---|
| `--color-ink` | `#274970` | `text-[#274970]` |
| `--color-ink-dark` | `#1f3557` | `:root { color: ... }` |
| `--color-muted` | `#8096b3` | `text-[#8096b3]` |
| `--color-muted-mid` | `#547094` | `text-[#547094]` |
| `--color-pink` | `#f08bb9` | `text-[#f08bb9]`, gradient stops |
| `--color-blue` | `#5b7faa` | `text-[#5b7faa]`, `bg-[#5b7faa]` |
| `--color-sky` | `#7dd9ff` | `rgba(125,217,255,...)` tints |
| `--color-success` | `#4ca9cf` | `text-[#4ca9cf]` |
| `--color-danger` | `#8d1d2c` | `bg-[#8d1d2c]`, `text-[#8d1d2c]` |
| `--color-warn` | `#b46088` | `text-[#b46088]` |
| `--color-border` | `rgba(106,147,198,0.12)` | 15+ occurrences of this exact value |
| `--color-surface-0` | `rgba(255,252,247,0.82)` | warm paper background |
| `--color-surface-1` | `rgba(255,255,255,0.76)` | workspace panel |
| `--color-surface-2` | `rgba(255,255,255,0.84)` | editable inputs |
| `--color-surface-3` | `rgba(255,255,255,0.92)` | inputs, dropdowns |

Opacity tints (e.g. `bg-sky/8`, `border-blue/45`) use Tailwind's built-in opacity modifier syntax — no extra tokens needed for those.

**Typography**

| Token | Value | Usage |
|---|---|---|
| `--font-serif` | `"Iowan Old Style","Palatino Linotype",serif` | `font-serif` on headings |
| `--font-mono` | `SFMono-Regular,Consolas,monospace` | `font-mono` on code/transcript |
| `--text-badge` | `0.5rem` | navbar language pip |
| `--text-caption` | `0.72rem` | platform labels, transcript headers |
| `--text-hint` | `0.82rem` | field hints, status chips |
| `--text-section` | `1.35rem` | panel/workspace titles |
| `--text-display` | `clamp(2rem,4vw,3.5rem)` | page headings |

All other sizes in the `0.82–0.94rem` cluster snap to Tailwind's built-in `text-sm` (0.875rem). `text-[0.8rem]` (eyebrow) snaps to `text-xs` (0.75rem).

**Shadows**

| Token | Value | Replaces |
|---|---|---|
| `--shadow-card` | `0 14px 28px rgba(116,148,194,0.07)` | `shadow-[0_14px_28px_...]` (4+ occurrences) |
| `--shadow-elevated` | `0 22px 44px rgba(116,148,194,0.12)` | jobs drawer |
| `--shadow-overlay` | `0 8px 32px rgba(0,0,0,0.12)` | identity panel |

**Border Radius** (semantic names, exact px preserved)

| Token | Value | Replaces | Used on |
|---|---|---|---|
| `--radius-icon` | `10px` | `rounded-[10px]` | nav icon buttons |
| `--radius-overlay` | `14px` | `rounded-[14px]` | identity panel, tooltips |
| `--radius-card-sm` | `18px` | `rounded-[18px]` | smaller cards/panels |
| `--radius-card` | `22px` | `rounded-[22px]` | list cards, app panels |
| `--radius-drawer` | `28px` | `rounded-[28px]` | jobs drawer |

**Z-index**

| Token | Value | Replaces |
|---|---|---|
| `--z-nav` | `100` | `z-[100]` |
| `--z-backdrop` | `199` | `z-[199]` |
| `--z-overlay` | `200` | `z-[200]` |

### Spacing Normalization

Arbitrary spacing values are mapped to Tailwind's built-in scale (1 unit = 4px) where they land on it:

| Arbitrary | Normalized | Note |
|---|---|---|
| `px-[14px]` | `px-3.5` | 3.5 × 4 = 14 ✓ |
| `py-[18px]` | `py-4.5` | 4.5 × 4 = 18 ✓ |
| `gap-[14px]` | `gap-3.5` | |
| `px-[10px]` | `px-2.5` | |
| `py-[11px]` | `py-[11px]` | keep — not on scale |
| `px-[3px]` | `px-[3px]` | keep — badge micro-spacing |

---

### `ui.ts` Restructuring

Constants are grouped into five named sections with header comments. Shared bases are extracted to reduce duplication. Every non-obvious class gets a one-line comment.

**Sections:**

1. **Surfaces & Layout** — `appPanelClass`, `workspacePanelClass`, `workspacePanelTitleClass`, `workspacePanelBodyClass`
2. **Typography** — `pageHeadingClass`, `sectionTitleClass`, `mutedTextClass`, `eyebrowClass`
3. **Form Fields** — `fieldClass`, `fieldLabelClass`, `fieldHintClass`, `inputClass` (derived from `inputBase`), `textareaClass`, `checkboxRowClass`
4. **Settings Layout** — `settingsFieldClass`, `settingsFieldMetaClass`, `settingsControlClass`, `settingsInputClass`, `settingsSelectClass`
5. **Buttons** — all four button classes derived from `buttonBase`
6. **Status** — `statusSuccessClass`, `statusErrorClass`, `statusChipClass()`

**Shared bases (unexported):**

```ts
const inputBase = "w-full rounded-2xl border border-border bg-surface-3 px-3.5 py-3 text-ink outline-none transition focus:border-blue/45 focus:ring-2 focus:ring-sky/18";

const buttonBase = "inline-flex items-center justify-center rounded-2xl px-4 py-[11px] transition disabled:cursor-not-allowed disabled:opacity-60";
```

`textareaClass`, `settingsInputClass`, and `settingsSelectClass` all extend `inputBase`. The four button variants all extend `buttonBase`.

**Comment policy:**
- Section header comments (`// ─── Section Name ─────`)
- One-line intent comment on non-obvious constants (eyebrow, settingsField layout, fluid heading)
- Gradient descriptions on button/logo gradients
- No comments on self-explanatory constants (`mutedTextClass`, `fieldClass`, etc.)

**What stays out of `ui.ts`:** One-off layout values that appear in exactly one component stay inline in that component with a comment. Examples:
- Jobs drawer: `w-[min(420px,calc(100vw-24px))]` — responsive drawer width capped at 420px
- AppFrame content: `px-[clamp(16px,10vw,160px)]` — fluid horizontal page padding
- Navbar height offset: `pt-[calc(52px+28px)]` — 52px navbar + 28px breathing room

---

## Phase B: Component Extraction

### Scope

Extract the most-reused UI patterns from class string constants into thin React wrapper components. Each component:
- Accepts a `className` prop for call-site overrides
- Forwards all standard HTML props via spread (`...rest`)
- Has no internal state
- Lives in `web/src/components/ui/`

### Components

**`<Button>`**

```tsx
// variant covers all four button styles; size defaults to md
<Button variant="primary" | "secondary" | "ghost" | "danger" size="md" | "sm" className? ...rest />
```

Replaces all `<button className={primaryButtonClass}>` patterns (21 call sites across 10 files).

`size="sm"` adds a smaller padding/text variant for icon-adjacent buttons.

**`<FormField>`**

```tsx
// Renders label + hint text wrapper; children is the control
<FormField label="Model" hint="Optional description" className? ...rest />
```

Replaces the repeated `fieldClass` / `fieldLabelClass` / `fieldHintClass` triple (used in every settings tab, source list, create form).

**`<Panel>`**

```tsx
// variant="app" | "workspace"; maps to appPanelClass / workspacePanelClass
<Panel variant="app" className? ...rest />
```

Replaces 8+ instances of `<div className={appPanelClass}>`.

**`<StatusChip>`**

```tsx
<StatusChip status="ok" | "error" | "unavailable" | "neutral">{children}</StatusChip>
```

Replaces `statusChipClass()` call sites (health panel, dependency rows).

### What does NOT become a component

- `eyebrowClass`, `sectionTitleClass`, `pageHeadingClass` — single-element, single call site per page. Keep as class constants.
- `settingsFieldClass` group — complex enough that extraction adds indirection without simplifying call sites. Keep in `ui.ts`.
- Anything used in only one place.

---

## File Changelist

### Phase A
| Action | File |
|---|---|
| Modify | `web/src/styles/app.css` — add `@theme` block |
| Modify | `web/src/lib/ui.ts` — replace arbitrary values, add grouping + comments |
| Modify | `web/src/components/layout/AppFrame.tsx` |
| Modify | `web/src/components/layout/IdentityPanel.tsx` |
| Modify | `web/src/components/jobs/JobsBadge.tsx` |
| Modify | `web/src/components/lists/ListGrid.tsx` |
| Modify | `web/src/components/sources/SourceDetail.tsx` |
| Modify | `web/src/pages/ListDetailPage.tsx` |
| Modify | `web/src/components/chat/ChatPanel.tsx` |

### Phase B
| Action | File |
|---|---|
| Create | `web/src/components/ui/Button.tsx` |
| Create | `web/src/components/ui/FormField.tsx` |
| Create | `web/src/components/ui/Panel.tsx` |
| Create | `web/src/components/ui/StatusChip.tsx` |
| Create | `web/src/components/ui/index.ts` — barrel export |
| Modify | All files currently using `primaryButtonClass`, `appPanelClass`, `fieldClass`, `statusChipClass` |
| Modify | `web/src/lib/ui.ts` — remove exported constants replaced by components |

---

## Testing

- `npm run lint` must pass after each phase (TypeScript + no unused exports)
- `npm run test` must pass after each phase (no regressions in existing tests)
- No new tests required for Phase A (pure class string changes, no logic)
- Phase B components get one smoke test each: renders without crashing, applies variant class, forwards className prop

---

## Success Criteria

- Zero `[rgba(...)]` or `[#hex]` arbitrary color values remaining in any component file
- Zero `shadow-[...]` arbitrary shadow values remaining
- Zero `rounded-[Npx]` arbitrary radius values remaining (except the three documented inline exceptions)
- `ui.ts` has section headers and comments on every non-obvious constant
- `npm run lint` and `npm run test` green after each phase
