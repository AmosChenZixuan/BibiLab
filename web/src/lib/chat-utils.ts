import { FIND_PASSAGES_TOOL_NAME, READ_SECTION_TOOL_NAME, translateOrFallback } from "@/lib/utils";

export type ContentBlock =
  | { type: "text"; text: string }
  | { type: "citation"; index: number; section_id: string; source_id: string;
      timestamp_start: number; chunk_ids: string[] }
  | { type: "paragraph_break" };

type ToolName = typeof FIND_PASSAGES_TOOL_NAME | typeof READ_SECTION_TOOL_NAME;

/** Args passed to `onOpenSource` when a citation chip is clicked.
 *  `sectionId` + `timestampStart` are the citation jump target (used by
 *  SourcesViewerMode to land on the cited section's tab). Hoisted to
 *  lib/chat-utils so the shape is shared between the chip call site
 *  (ChatPanel) and the page-level handler (ListDetailPage). */
export type OpenSourceOpts = {
  highlightChunks?: string[];
  sectionId?: string;
  timestampStart?: number;
};

/** Buffered message piped from a non-chat UI trigger (digest keyword,
 *  mindmap node click) into the always-mounted ChatPanel. ChatPanel
 *  drains via its `pendingMessage` prop effect and always acks
 *  (`onPendingMessageConsumed`) so the prop is cleared whether the send
 *  is dispatched or rejected. The `nonce` is an effect re-fire trigger:
 *  identical clicks (same text) need a fresh nonce to re-run the
 *  drain effect. `sourceIds` is an optional one-shot override so a
 *  caller can scope a single send to a non-current selection without
 *  mutating the page-level selection state. */
export type PendingChatMessage = {
  text: string;
  nonce: number;
  sourceIds?: string[];
};

/** Page-level handler fired when a mindmap node is clicked. `sourceIds`
 *  is the artifact's persistent `source_ids` (MindMapBlock has no
 *  knowledge of sources; ArtifactViewer injects them). The page turns
 *  the structured data into a localized message and queues it via
 *  `PendingChatMessage`. */
export type MindMapAskInChat = (
  topic: string,
  parentTopic: string | null,
  sourceIds: string[],
) => void;

// NB: `section_id` on SectionCoverage / RetrievalChunk below is the integer
// sections.id serialized by the backend — declared string here to mirror the
// payload, but it arrives as a number at runtime. These fields are display-only
// (the ledger reads section_seq / .length, never section_id), so they are NOT
// coerced. Anyone adding a strict-equality match on them must coerce first —
// see coerceContentBlock for the citation path that does.

/** A surfaced section in a find_passages result. */
type SectionCoverage = {
  section_id: string;
  source_id: string;
  source_title: string;
  seq: number;
  timestamp_start: number;
  timestamp_end: number;
};

/** Single chunk in the persisted context[] array.
 *  For outline-only entries (facet match, no chunks), `chunk_id` and
 *  `rerank_score` are null and `preview` carries the section summary.
 *  `section_id` is always populated. */
type RetrievalChunk = {
  chunk_id: string | null;
  citation_index: number;
  section_id: string;
  section_seq: number;
  source_id: string;
  source_title: string;
  timestamp_start: number;
  timestamp_end: number;
  rerank_score: number | null;
  preview: string;
};
/** Echo of the LLM-extracted facet predicate + deterministic match outcome.
 * `no_match` is true iff a predicate was given AND zero sources matched (the
 * backend then fails open to the full pre-facet pool, surfaced to the user as
 * a non-blocking hint in the retrieval ledger). */
type FacetScope = {
  sequence_number: number | null;
  season_number: number | null;
  matched_count: number | null;
  no_match: boolean;
};
/** section_coverage lists only sections whose [N] actually appeared in the assistant text. */
export type RetrievalCall = {
  query: string;
  tool_name: ToolName;
  candidates_evaluated: number;
  sources_with_hits: number;
  sources_total: number;
  section_coverage: SectionCoverage[];
  // context[] is absent on the streaming tool_result payload;
  // reconstructed only in persisted metadata.rag. For read_section,
  // context is always [] (continuous transcript, not a locator result).
  context?: RetrievalChunk[];
  reranked: boolean;
  scoped_pool_size: number;
  facet_scope?: FacetScope;
  // read_section rows only — find_passages uses section_coverage
  section_id?: string;
  source_id?: string;
  source_title?: string;
};
export type RagMetadata = { calls: RetrievalCall[] };
export type PendingRagCall = {
  id: string;
  query: string;
  tool_name: ToolName;
};

/** Coerce a raw SSE `citation` event into the `ContentBlock` citation shape.
 *
 * The backend's `CitationRegistryEntry.section_id` is the INTEGER
 * `sections.id` (the row PK). The value flows from `get_sections`'s
 * `r["id"]` into the chunk loop in `chat_tools.py` (`_alloc_section`
 * is just the funnel that stores it on the entry), then serializes
 * over SSE as a JSON number. The FE's `ContentBlock.citation.
 * section_id` type declares `string`, and the citation-jump matcher
 * in `SourcesViewerMode.resolveTargetIdx` does strict equality
 * against `SourceSection.section_id` (also `string`). A number on
 * one side would silently fall through to the `timestampStart`
 * branch, landing the reader on the wrong section.
 *
 * Normalize the field to string here so the type contract and the
 * jump both work without a backend serialization change. Defaults
 * to `""` when missing so the caller's `if (target.sectionId)` falsy
 * check correctly skips the sectionId branch.
 */
export function coerceCitationEvent(raw: unknown): ContentBlock {
  const e = raw as { type?: string; index?: number; section_id?: unknown; source_id?: unknown; timestamp_start?: unknown; chunk_ids?: unknown };
  if (e?.type !== "citation") {
    return { type: "paragraph_break" };
  }
  return {
    type: "citation",
    index: Number(e.index ?? 0),
    section_id: e.section_id != null ? String(e.section_id) : "",
    source_id: String(e.source_id ?? ""),
    timestamp_start: Number(e.timestamp_start ?? 0),
    chunk_ids: Array.isArray(e.chunk_ids) ? e.chunk_ids.map(String) : [],
  };
}

/** Coerce a persisted/streamed `ContentBlock` so a citation's `section_id`
 *  is a string. Citation blocks reach the FE from two paths — the live SSE
 *  `citation` event (handled by `coerceCitationEvent`) and the persisted
 *  `metadata.content_blocks` on history reload — and both carry the integer
 *  `sections.id`. Text / paragraph_break blocks pass through unchanged. */
export function coerceContentBlock(raw: unknown): ContentBlock {
  const b = raw as { type?: string; text?: unknown };
  if (b?.type === "citation") return coerceCitationEvent(raw);
  if (b?.type === "text") return { type: "text", text: String(b.text ?? "") };
  return { type: "paragraph_break" };
}

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

/** Human sentence for a fail-open facet no-match. Empty facet set →
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
