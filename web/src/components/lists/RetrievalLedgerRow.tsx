import { useState } from "react";
import { Search, RefreshCw, Loader2, AlertTriangle, ChevronRight } from "lucide-react";
import { useLanguage } from "@/app/LanguageContext";
import { formatMediaTimestamp, type RetrievalCall, type PendingRagCall, type PendingMetadataCall } from "@/lib/chat-utils";

export type RowVariant = "default" | "empty" | "reused" | "pending";

interface RetrievalLedgerRowProps {
  variant: RowVariant;
  call?: RetrievalCall;
  pending?: PendingRagCall | PendingMetadataCall;
  /** While the message streams, context[] is incomplete — render collapsed
   * status only (no expand) until the final `rag` event arrives. */
  streaming?: boolean;
}

function ScopeDisplay({ call, t }: { call: RetrievalCall; t: (key: string, params?: Record<string, string | number>) => string }) {
  const { scope_choice, excluded_count, scoped_pool_size, sources_total } = call;
  if (scope_choice === "exclude") {
    return <>{t("chat.ledger.scope.exclude", { count: excluded_count ?? 0, total: sources_total })}</>;
  }
  if (scope_choice === "whitelist") {
    return <>{t("chat.ledger.scope.whitelist", { count: scoped_pool_size, total: sources_total })}</>;
  }
  return <>{t("chat.ledger.scope.none", { total: sources_total })}</>;
}

function ModeDisplay({ expected_hits, t }: { expected_hits: RetrievalCall["expected_hits"]; t: (key: string) => string }) {
  const key = expected_hits ? `chat.ledger.mode.${expected_hits}` : "chat.ledger.modeUnknown";
  return <>{t(key)}</>;
}

export function RetrievalLedgerRow({ variant, call, pending, streaming = false }: RetrievalLedgerRowProps) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(false);

  if (variant === "reused") {
    return (
      <div className="flex items-center gap-1.5 py-0.5 text-xs text-muted opacity-60">
        <RefreshCw size={12} className="shrink-0" />
        <span>{t("chat.ledger.summaryReused")}</span>
      </div>
    );
  }

  if (variant === "pending" && pending) {
    const isRag = "expected_hits" in pending;
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

  if (variant === "empty") {
    const summaryText = t("chat.ledger.summaryEmpty", { dropped: call.dropped_by_gate });
    if (streaming) {
      return (
        <div className="flex w-full items-center gap-1.5 text-xs text-amber">
          <AlertTriangle size={12} className="shrink-0" />
          <span className="truncate">{summaryText}</span>
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
              <span className="font-medium text-ink"><ModeDisplay expected_hits={call.expected_hits} t={t} /></span>
            </span>
            <span>
              <span className="text-muted">{t("chat.ledger.field.scope")} </span>
              <span className="font-medium text-ink"><ScopeDisplay call={call} t={t} /></span>
            </span>
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
