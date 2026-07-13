export const FIND_PASSAGES_TOOL_NAME = "find_passages";
export const READ_SECTION_TOOL_NAME = "read_section";

export const LANG_STORAGE_KEY = "bibilab-lang";

export type Lang = "en" | "zh";

export function getUiLang(): Lang {
  return localStorage.getItem(LANG_STORAGE_KEY) === "zh" ? "zh" : "en";
}

export function setUiLang(next: Lang): void {
  localStorage.setItem(LANG_STORAGE_KEY, next);
}

export function translateOrFallback(t: (key: string) => string, key: string, fallback: string): string {
  const translated = t(key);
  return translated !== key ? translated : fallback;
}

export const localCoverUrl = (sourceId: string): string => `/api/sources/${sourceId}/cover`;

export const proxyCoverUrl = (url: string): string =>
  `/api/proxy/cover?url=${encodeURIComponent(url)}`;

const BILIBILI_DOMAINS = ["bilibili.com", "b23.tv"];

export function isBilibiliUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return BILIBILI_DOMAINS.some((d) => host === d || host.endsWith(`.${d}`));
  } catch {
    return false;
  }
}

export function formatBundleSize(sizeMb: number | null): string {
  if (sizeMb == null) return "—";
  if (sizeMb >= 1000) return `${(sizeMb / 1000).toFixed(1)} GB`;
  return `${sizeMb} MB`;
}
