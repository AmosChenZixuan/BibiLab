export type ContentBlock =
  | { type: "text"; text: string }
  | { type: "citation"; index: number; source_id: string; chunk_ids: string[] };

export type ToolResult = { artifact_id: string; job_id?: string; name: string; type: string };
export interface ToolCallData {
  name: string;
  result: ToolResult;
}
export type SearchMode = "factual" | "breadth" | "analytical";
export type RagSource = { source_id: string; video_id: string; title: string };
export type RagMetadata = {
  search_mode: SearchMode;
  candidates_evaluated: number;
  sources_with_hits: number;
  sources_total: number;
  source_coverage: RagSource[];
};

export function formatDurationHuman(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
  return `${m}m`;
}

export function formatSubtitle(
  t: (key: string, params?: Record<string, string | number>) => string,
  sourceCount: number,
  totalSeconds: number,
): string {
  const key = sourceCount === 1 ? "chat.subtitle.templateSingular" : "chat.subtitle.templatePlural";
  return t(key, { count: sourceCount, duration: formatDurationHuman(totalSeconds) });
}

const LEGACY_CITATION_RE = /\[([^\]]+?) @ (\d+)s-(\d+)s\]/g;

export function stripLegacyTokens(text: string): string {
  return text.replace(LEGACY_CITATION_RE, "");
}

export function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function autoResize(ta: HTMLTextAreaElement) {
  if (!ta.value) {
    ta.style.height = "auto";
    ta.style.overflowY = "hidden";
  } else {
    const maxHeight = 200;
    ta.style.height = "0";
    ta.style.height = `${Math.min(ta.scrollHeight, maxHeight)}px`;
    ta.style.overflowY = ta.scrollHeight > maxHeight ? "auto" : "hidden";
  }
}
