import { useState } from "react";
import { Search, BookOpen, Loader2, AlertTriangle, ChevronRight } from "lucide-react";
import { useLanguage } from "@/app/LanguageContext";
import { formatMediaTimestamp, facetNoMatchHint, type RetrievalCall, type PendingRagCall } from "@/lib/chat-utils";
import { READ_SOURCE_TOOL_NAME, type ToolDisplayConfig } from "@/lib/tool-display";

interface ToolLedgerRowProps {
  config: ToolDisplayConfig;
  call?: RetrievalCall;
  pending?: PendingRagCall;
  streaming?: boolean;
}

export function ToolLedgerRow({ config, call, pending, streaming = false }: ToolLedgerRowProps) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(false);

  if (pending) {
    if (pending.tool_name === READ_SOURCE_TOOL_NAME) {
      return (
        <div className="flex items-center gap-1.5 py-0.5 text-xs text-muted opacity-70">
          <Loader2 size={12} className="animate-spin shrink-0" />
          <span>{t("chat.ledger.readSourceLabel")}</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-1.5 py-0.5 text-xs text-muted opacity-70">
        <Loader2 size={12} className="animate-spin shrink-0" />
        <span>{t("chat.ledger.summaryPending")}</span>
      </div>
    );
  }

  if (!call) return null;

  // read_source: compact "read in full" chip — no per-chunk expand (context is always [])
  if (call.tool_name === READ_SOURCE_TOOL_NAME) {
    return (
      <div className="flex w-full items-center gap-1.5 text-xs text-muted">
        <BookOpen size={12} className="shrink-0 self-center opacity-70" aria-hidden />
        <span>{t("chat.ledger.readSourceLabel")}</span>
        {call.source_title && <span className="min-w-0 truncate font-medium text-ink">{call.source_title}</span>}
      </div>
    );
  }

  // find_passages (locator) call
  const source_coverage = call.source_coverage ?? [];
  const context = call.context ?? [];

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

  // Default variant: normal search row
  const citedCount = context.length;
  const sourceSummary = t("chat.ledger.sourceSummary", { sources: source_coverage.length });
  if (streaming) {
    return (
      <div className="flex w-full items-baseline gap-1.5 text-xs text-muted">
        <Search size={12} className="shrink-0 self-center opacity-70" />
        <span className="min-w-0 truncate font-mono">{call.query}</span>
        <span className="shrink-0 opacity-40">·</span>
        <span className="shrink-0">{sourceSummary}</span>
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
        <span className="shrink-0">{sourceSummary}</span>
        {citedCount > 0 && (
            <>
              <span className="shrink-0 opacity-40">·</span>
              <div className="shrink-0">{t("chat.ledger.citedChunks", { n: citedCount })}</div>
            </>
          )}
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
              <span className="text-muted">{t("chat.ledger.field.scope")} </span>
              <span className="font-medium text-ink">{
                (() => {
                  const fs = call.facet_scope;
                  return !fs || fs.no_match || fs.matched_count == null
                    ? t("chat.ledger.scope.none", { total: call.sources_total })
                    : t("chat.ledger.scope.scoped", { matched: fs.matched_count, total: call.sources_total });
                })()
              }</span>
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
