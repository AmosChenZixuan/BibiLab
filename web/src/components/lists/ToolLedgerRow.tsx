import { useState } from "react";
import { Search, Loader2, AlertTriangle, ChevronRight } from "lucide-react";
import { useLanguage } from "@/app/LanguageContext";
import { formatMediaTimestamp, facetNoMatchHint, type RetrievalCall, type MetadataCall, type PendingRagCall, type PendingMetadataCall } from "@/lib/chat-utils";
import { METADATA_TOOL_NAME, type ToolDisplayConfig } from "@/lib/tool-display";

interface ToolLedgerRowProps {
  config: ToolDisplayConfig;
  call?: RetrievalCall | MetadataCall;
  pending?: PendingRagCall | PendingMetadataCall;
  streaming?: boolean;
}

function isMetadataCall(call: RetrievalCall | MetadataCall): call is MetadataCall {
  return (call as MetadataCall).name === METADATA_TOOL_NAME;
}

function ModeDisplay({ mode, t }: { mode: RetrievalCall["mode"]; t: (key: string) => string }) {
  const key = mode ? `chat.ledger.mode.${mode}` : "chat.ledger.modeUnknown";
  return <>{t(key)}</>;
}

export function ToolLedgerRow({ config, call, pending, streaming = false }: ToolLedgerRowProps) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(false);

  if (pending) {
    const isRag = "mode" in pending && !("query_type" in pending);
    const label = isRag
      ? t("chat.ledger.summaryPending")
      : t("chat.ledger.metadataPending");
    return (
      <div className="flex items-center gap-1.5 py-0.5 text-xs text-muted opacity-70">
        <Loader2 size={12} className="animate-spin shrink-0" />
        <span>{label}</span>
      </div>
    );
  }

  if (!call) return null;

  // Metadata call: icon + label, expandable to raw JSON
  if (isMetadataCall(call)) {
    const resultJson = JSON.stringify(call.result, null, 2);
    if (streaming) {
      return (
        <div className="flex w-full items-center gap-1.5 text-xs text-muted">
          <config.icon size={12} className="shrink-0 self-center opacity-70" />
          <span>{t(config.labelKey ?? "")}</span>
        </div>
      );
    }
    return (
      <div className="w-full overflow-hidden text-xs text-muted">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex w-full items-center gap-1.5 border-none bg-transparent p-0 text-xs text-muted cursor-pointer hover:text-ink"
          aria-expanded={expanded}
          aria-label={t("chat.ledger.ariaToggle")}
        >
          <config.icon size={12} className="shrink-0 self-center opacity-70" />
          <span>{t(config.labelKey ?? "")}</span>
          <ChevronRight size={12} className={`ml-auto shrink-0 self-center transition-transform ${expanded ? "rotate-90" : ""}`} />
        </button>
        {expanded && (
          <div className="mt-1.5 border-t border-border pt-1.5">
            <pre className="m-0 whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-ink">{resultJson}</pre>
          </div>
        )}
      </div>
    );
  }

  // Search call (retrieve / survey / retrieve_scoped)
  const ragCall = call as RetrievalCall;
  const source_coverage = ragCall.source_coverage ?? [];
  const context = ragCall.context ?? [];

  const facetHint = ragCall.facet_scope?.no_match ? facetNoMatchHint(t, ragCall.facet_scope) : null;
  const facetIcon = facetHint ? (
    <AlertTriangle size={12} className="shrink-0 self-center text-amber" aria-label={facetHint} />
  ) : null;
  const facetDetailLine = facetHint ? (
    <span>
      <span className="text-muted">{t("chat.ledger.facetNoMatchLabel")} </span>
      <span className="font-medium text-amber">{facetHint}</span>
    </span>
  ) : null;

  // Empty variant: all chunks gated out
  if (ragCall.context != null && ragCall.context.length === 0 && ragCall.dropped_by_gate > 0) {
    const summaryText = t("chat.ledger.summaryEmpty", { dropped: ragCall.dropped_by_gate });
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
              <span className="font-medium text-ink truncate">{ragCall.query}</span>
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

  // Default variant: normal search row
  const citedCount = context.length;
  const sourceSummary = t("chat.ledger.sourceSummary", { sources: source_coverage.length });
  if (streaming) {
    return (
      <div className="flex w-full items-baseline gap-1.5 text-xs text-muted">
        <Search size={12} className="shrink-0 self-center opacity-70" />
        <span className="min-w-0 truncate font-mono">{ragCall.query}</span>
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
        <span className="min-w-0 truncate font-mono">{ragCall.query}</span>
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
              <span className="font-medium text-ink">{ragCall.query}</span>
            </span>
            <span>
              <span className="text-muted">{t("chat.ledger.field.mode")} </span>
              <span className="font-medium text-ink"><ModeDisplay mode={ragCall.mode} t={t} /></span>
            </span>
            <span>
              <span className="text-muted">{t("chat.ledger.field.scope")} </span>
              <span className="font-medium text-ink">{
                (() => {
                  const fs = ragCall.facet_scope;
                  return !fs || fs.no_match || fs.matched_count == null
                    ? t("chat.ledger.scope.none", { total: ragCall.sources_total })
                    : t("chat.ledger.scope.scoped", { matched: fs.matched_count, total: ragCall.sources_total });
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
