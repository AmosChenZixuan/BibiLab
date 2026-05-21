import { useState } from "react";
import { Search, Loader2, AlertTriangle, ChevronRight } from "lucide-react";
import { useLanguage } from "@/app/LanguageContext";
import { formatMediaTimestamp, facetNoMatchHint, type RetrievalCall, type PendingRagCall, type PendingMetadataCall } from "@/lib/chat-utils";

export type RowVariant = "default" | "empty" | "pending";

interface RetrievalLedgerRowProps {
  variant: RowVariant;
  call?: RetrievalCall;
  pending?: PendingRagCall | PendingMetadataCall;
  /** While the message streams, context[] is incomplete — render collapsed
   * status only (no expand) until the final `rag` event arrives. */
  streaming?: boolean;
}

function ModeDisplay({ mode, t }: { mode: RetrievalCall["mode"]; t: (key: string) => string }) {
  const key = mode ? `chat.ledger.mode.${mode}` : "chat.ledger.modeUnknown";
  return <>{t(key)}</>;
}

export function RetrievalLedgerRow({ variant, call, pending, streaming = false }: RetrievalLedgerRowProps) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(false);

  if (variant === "pending" && pending) {
    const isRag = "mode" in pending;
    const label = isRag
      ? t("chat.ledger.summaryPending")
      : t(`chat.ledger.metadataPending.${(pending as PendingMetadataCall).query_type}`, {});
    return (
      <div className="flex items-center gap-1.5 py-0.5 text-xs text-muted opacity-70">
        <Loader2 size={12} className="animate-spin shrink-0" />
        <span>{label || (pending as PendingMetadataCall).query_type}</span>
      </div>
    );
  }

  if (!call) return null;

  const source_coverage = call.source_coverage ?? [];
  const context = call.context ?? [];

  // #319 fail-open visibility. Hoisted so every collapsed/streaming/expanded
  // branch shares one definition.
  const facetHint = call.facet_scope?.no_match ? facetNoMatchHint(t, call.facet_scope) : null;
  const facetIcon = facetHint ? (
    <AlertTriangle size={12} className="shrink-0 self-center text-amber" aria-label={facetHint} />
  ) : null;
  const facetDetailLine = facetHint ? (
    <span>
      <span className="text-muted">{t("chat.ledger.facetNoMatchLabel")} </span>
      <span className="font-medium text-amber">{facetHint}</span>
    </span>
  ) : null;

  if (variant === "empty") {
    const summaryText = t("chat.ledger.summaryEmpty", { dropped: call.dropped_by_gate });
    if (streaming) {
      return (
        <div className="flex w-full items-center gap-1.5 text-xs text-amber">
          <AlertTriangle size={12} className="shrink-0" />
          <span className="truncate">{summaryText}</span>
          {facetIcon}
        </div>
      );
    }
    return (
      <div className="w-full overflow-hidden text-xs text-amber">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex w-full items-center gap-1.5 border-none bg-transparent p-0 text-xs text-amber cursor-pointer"
          aria-expanded={expanded}
          aria-label={t("chat.ledger.ariaToggle")}
        >
          <AlertTriangle size={12} className="shrink-0" />
          <span className="truncate">{summaryText}</span>
          {facetIcon}
          <ChevronRight size={12} className={`ml-auto shrink-0 transition-transform ${expanded ? "rotate-90" : ""}`} />
        </button>
        {expanded && (
          <div className="mt-1.5 flex flex-col gap-1 border-t border-border pt-1.5">
            <div className="flex gap-2">
              <span className="text-muted">{t("chat.ledger.field.query")}</span>
              <span className="font-medium text-ink truncate">{call.query}</span>
            </div>
            <div className="flex gap-2">
              <span className="text-muted">{t("chat.ledger.field.result")}</span>
              <span className="font-medium text-ink">{summaryText}</span>
            </div>
            {facetDetailLine}
          </div>
        )}
      </div>
    );
  }

  // Streaming payload omits context[]; persisted context is one entry per
  // cited source, so fall back to source_coverage length for a consistent
  // count before/after refresh.
  const chunkCount = call.context?.length ?? source_coverage.length;
  const summaryText = t("chat.ledger.summary", { chunks: chunkCount, sources: source_coverage.length });
  if (streaming) {
    return (
      <div className="flex w-full items-baseline gap-1.5 text-xs text-muted">
        <Search size={12} className="shrink-0 self-center opacity-70" />
        <span className="min-w-0 truncate font-mono">{call.query}</span>
        <span className="shrink-0 opacity-40">·</span>
        <span className="shrink-0">{summaryText}</span>
        {facetIcon}
      </div>
    );
  }
  return (
    <div className="w-full overflow-hidden text-xs text-muted">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-baseline gap-1.5 border-none bg-transparent p-0 text-xs text-muted cursor-pointer hover:text-ink"
        aria-expanded={expanded}
        aria-label={t("chat.ledger.ariaToggle")}
      >
        <Search size={12} className="shrink-0 self-center opacity-70" />
        <span className="min-w-0 truncate font-mono">{call.query}</span>
        <span className="shrink-0 opacity-40">·</span>
        <span className="shrink-0">{summaryText}</span>
        {facetIcon}
        <ChevronRight size={12} className={`ml-auto shrink-0 self-center transition-transform ${expanded ? "rotate-90" : ""}`} />
      </button>

      {expanded && (
        <div className="mt-1.5 flex flex-col gap-2 border-t border-border pt-1.5">
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            <span>
              <span className="text-muted">{t("chat.ledger.field.query")} </span>
              <span className="font-medium text-ink">{call.query}</span>
            </span>
            <span>
              <span className="text-muted">{t("chat.ledger.field.mode")} </span>
              <span className="font-medium text-ink"><ModeDisplay mode={call.mode} t={t} /></span>
            </span>
            <span>
              <span className="text-muted">{t("chat.ledger.field.scope")} </span>
              <span className="font-medium text-ink">{t("chat.ledger.scope.none", { total: call.sources_total })}</span>
            </span>
            {facetDetailLine}
          </div>

          {context.length > 0 && (
            <div className="flex flex-col gap-1.5 border-l border-border pl-3">
              {context.map((chunk) => (
                <div key={chunk.chunk_id} className="flex min-w-0 flex-col">
                  <div className="flex min-w-0 items-baseline gap-1.5">
                    <span className="shrink-0 font-mono text-blue">[{chunk.citation_index}]</span>
                    <span className="min-w-0 truncate text-ink">{chunk.source_title}</span>
                    <span className="ml-auto shrink-0 font-mono text-muted">
                      {formatMediaTimestamp(chunk.timestamp_start)}–{formatMediaTimestamp(chunk.timestamp_end)} · {chunk.rerank_score.toFixed(2)}
                    </span>
                  </div>
                  <div className="mt-0.5 text-muted line-clamp-2" title={chunk.preview}>
                    &ldquo;{chunk.preview}&rdquo;
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
