// ─── Surfaces & Layout ────────────────────────────────────────────────────────

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

export const fieldLabelClass = "text-sm font-semibold";
export const fieldHintClass  = "text-sm leading-5 text-muted";

// Base shared by inputClass, textareaClass, settingsInputClass
// (settingsSelectClass uses rounded-xl instead of rounded-2xl — intentional design difference)
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

// ─── Status ───────────────────────────────────────────────────────────────────

export const statusSuccessClass = "m-0 text-sm text-success";
export const statusErrorClass   = "m-0 text-sm text-danger";
