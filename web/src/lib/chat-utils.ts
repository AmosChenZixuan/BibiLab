import { translateOrFallback } from "@/lib/utils";

export type ContentBlock =
  | { type: "text"; text: string }
  | { type: "citation"; index: number; source_id: string; chunk_ids: string[] }
  | { type: "paragraph_break" };

export type ToolName = "find_passages" | "read_source";
export type RagSource = { source_id: string; title: string };
/** Single chunk in the persisted context[] array. */
export type RetrievalChunk = {
  chunk_id: string;
  citation_index: number;
  source_id: string;
  source_title: string;
  timestamp_start: number;
  timestamp_end: number;
  rerank_score: number;
  preview: string;
};
/** Echo of the LLM-extracted facet predicate + deterministic match outcome (#309).
 * `no_match` is true iff a predicate was given AND zero sources matched (the
 * backend then fails open to the full pre-facet pool — surfaced by #319). */
export type FacetScope = {
  sequence_number: number | null;
  season_number: number | null;
  matched_count: number | null;
  no_match: boolean;
};
/** source_coverage lists only sources whose [N] actually appeared in the assistant text. */
export type RetrievalCall = {
  query: string;
  tool_name: ToolName;
  candidates_evaluated: number;
  sources_with_hits: number;
  sources_total: number;
  source_coverage: RagSource[];
  // context[] is absent on the streaming tool_result payload;
  // reconstructed only in persisted metadata.rag. For read_source,
  // context is always [] (continuous transcript, not a locator result).
  context?: RetrievalChunk[];
  reranked: boolean;
  scoped_pool_size: number;
  facet_scope?: FacetScope;
  // read_source rows only — find_passages uses source_coverage
  source_id?: string;
  source_title?: string;
};
export type RagMetadata = { calls: RetrievalCall[] };
export type PendingRagCall = {
  id: string;
  query: string;
  tool_name: ToolName;
};

export function formatDurationHuman(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
  return `${m}m`;
}

export function formatSubtitle(t: (key: string, params?: Record<string, string | number>) => string, sourceCount: number, totalSeconds: number): string {
  const key = sourceCount === 1 ? "chat.subtitle.templateSingular" : "chat.subtitle.templatePlural";
  return t(key, {
    count: sourceCount,
    duration: formatDurationHuman(totalSeconds),
  });
}

/** Human sentence for a fail-open facet no-match (#319). Empty facet set →
 * generic copy (backend contract says no_match implies a predicate existed,
 * but stay defensive). */
export function facetNoMatchHint(
  t: (key: string, params?: Record<string, string | number>) => string,
  scope: FacetScope,
): string {
  const parts: string[] = [];
  if (scope.sequence_number != null) parts.push(t("chat.ledger.facet.sequence", { n: scope.sequence_number }));
  if (scope.season_number != null) parts.push(t("chat.ledger.facet.season", { n: scope.season_number }));
  if (parts.length === 0) return t("chat.ledger.facetNoMatchGeneric");
  return t("chat.ledger.facetNoMatch", { facets: parts.join(", ") });
}

const LEGACY_CITATION_RE = /\[([^\]]+?) @ (\d+)s-(\d+)s\]/g;

export function stripLegacyTokens(text: string): string {
  return text.replace(LEGACY_CITATION_RE, "");
}

export function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function formatMediaTimestamp(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.round(seconds % 60);
  const pad = (n: number) => n.toString().padStart(2, "0");
  if (h > 0) return `${h}:${pad(m)}:${pad(s)}`;
  return `${m}:${pad(s)}`;
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

export function getErrorLabel(error: string, t: (key: string) => string): string {
  return translateOrFallback(t, `chat.errors.${error}`, error);
}
