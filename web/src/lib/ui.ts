export const appPanelClass =
  "rounded-[22px] border border-[rgba(106,147,198,0.12)] bg-[rgba(255,252,247,0.82)] p-5 shadow-[0_14px_28px_rgba(116,148,194,0.07)]";

export const workspacePanelClass =
  "overflow-hidden rounded-3xl border border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.76)] shadow-[0_14px_28px_rgba(116,148,194,0.07)]";

export const workspacePanelTitleClass =
  'm-0 border-b border-[rgba(106,147,198,0.12)] px-5 py-[18px] font-["Iowan_Old_Style","Palatino_Linotype",serif] text-[1.35rem]';

export const workspacePanelBodyClass = "grid gap-4 px-5 py-[18px] pb-5";

export const pageHeadingClass =
  'm-0 mb-2 font-["Iowan_Old_Style","Palatino_Linotype",serif] text-[clamp(2rem,4vw,3.5rem)] leading-[0.95]';

export const sectionTitleClass =
  'm-0 font-["Iowan_Old_Style","Palatino_Linotype",serif] text-2xl';

export const mutedTextClass = "m-0 text-[#8096b3]";

export const eyebrowClass =
  "text-[0.8rem] uppercase tracking-[0.14em] text-[#f08bb9]";

export const fieldClass = "grid gap-1.5";
export const fieldLabelClass = "text-[0.92rem] font-semibold";
export const fieldHintClass = "text-[0.82rem] leading-5 text-[#8096b3]";
export const inputClass =
  "w-full rounded-2xl border border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.92)] px-[14px] py-3 text-[#274970] outline-none transition focus:border-[rgba(91,127,170,0.45)] focus:ring-2 focus:ring-[rgba(125,217,255,0.18)]";
export const textareaClass = `${inputClass} min-h-[96px] resize-y`;
export const checkboxRowClass = "inline-flex items-center gap-2.5";
export const settingsFieldClass =
  "flex flex-wrap items-start gap-x-5 gap-y-2 bg-[rgba(255,255,255,0.36)] px-4 py-3";
export const settingsFieldMetaClass = "min-w-[190px] flex-1 basis-[240px] grid gap-1";
export const settingsControlClass = "w-full min-w-[220px] flex-none md:w-[320px]";
export const settingsInputClass = `${inputClass} h-11 min-h-11 px-3 py-2.5`;
export const settingsSelectClass =
  "w-full min-w-[220px] flex-none rounded-xl border border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.92)] px-3 py-2.5 text-[#274970] outline-none transition focus:border-[rgba(91,127,170,0.45)] focus:ring-2 focus:ring-[rgba(125,217,255,0.18)] h-11 min-h-11 md:w-[320px]";

export const primaryButtonClass =
  "inline-flex items-center justify-center rounded-2xl border border-transparent bg-[linear-gradient(135deg,#f08bb9_0%,#5b7faa_100%)] px-4 py-[11px] text-[#fff8f1] transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60";
export const secondaryButtonClass =
  "inline-flex items-center justify-center rounded-2xl border border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.92)] px-4 py-[11px] text-[#274970] transition hover:bg-[rgba(255,255,255,1)] disabled:cursor-not-allowed disabled:opacity-60";
export const ghostButtonClass =
  "inline-flex items-center justify-center rounded-2xl border border-[rgba(91,127,170,0.18)] bg-transparent px-4 py-[11px] text-[#5b7faa] transition hover:bg-[rgba(125,217,255,0.08)] disabled:cursor-not-allowed disabled:opacity-60";
export const dangerButtonClass =
  "inline-flex items-center justify-center rounded-2xl border border-transparent bg-[#8d1d2c] px-4 py-[11px] text-[#fff7f6] transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60";

export const statusSuccessClass = "m-0 text-[0.94rem] text-[#4ca9cf]";
export const statusErrorClass = "m-0 text-[0.94rem] text-[#8d1d2c]";

export function statusChipClass(status: "ok" | "error" | "unavailable" | "neutral" = "neutral") {
  const base =
    "inline-flex items-center rounded-full border border-[rgba(106,147,198,0.12)] px-[10px] py-1.5 text-[0.82rem] capitalize";
  if (status === "ok") return `${base} text-[#4ca9cf]`;
  if (status === "error") return `${base} text-[#8d1d2c]`;
  if (status === "unavailable") return `${base} text-[#b46088]`;
  return `${base} text-[#5b7faa]`;
}
