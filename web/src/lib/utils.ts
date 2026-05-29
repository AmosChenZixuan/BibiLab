export const LANG_STORAGE_KEY = "bibilab-lang";

export function translateOrFallback(t: (key: string) => string, key: string, fallback: string): string {
  const translated = t(key);
  return translated !== key ? translated : fallback;
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function formatBundleSize(sizeMb: number | null): string {
  if (sizeMb == null) return "—";
  if (sizeMb >= 1000) return `${(sizeMb / 1000).toFixed(1)} GB`;
  return `${sizeMb} MB`;
}
