import { useState } from "react";
import { Loader2 } from "lucide-react";
import { useLanguage } from "@/app/LanguageContext";
import type { RetrievalCall, PendingRagCall, PendingMetadataCall } from "@/lib/chat-utils";

export type RowVariant = "default" | "empty" | "reused" | "pending";

interface RetrievalLedgerRowProps {
  variant: RowVariant;
  call?: RetrievalCall;
  pending?: PendingRagCall | PendingMetadataCall;
}

function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
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

export function RetrievalLedgerRow({ variant, call, pending }: RetrievalLedgerRowProps) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(false);

  // ----- reused -----
  if (variant === "reused") {
    return (
      <div className="bg-muted/20 border border-muted/30 opacity-70 px-2 py-0.5 rounded text-xs text-ink flex items-center gap-1">
        <span>⟲</span>
        <span>{t("chat.ledger.summaryReused")}</span>
      </div>
    );
  }

  // ----- pending -----
  if (variant === "pending" && pending) {
    const isRag = "expected_hits" in pending;
    const label = isRag
      ? t("chat.ledger.summaryPending")
      : t(`chat.ledger.metadataPending.${(pending as PendingMetadataCall).query_type}`, {});
    return (
      <div className="bg-sky/5 animate-pulse px-2 py-0.5 rounded text-xs text-blue flex items-center gap-1">
        <Loader2 size={11} className="animate-spin" />
        <span>{label || (pending as PendingMetadataCall).query_type}</span>
      </div>
    );
  }

  // ----- default / empty -----
  if (!call) return null;

  const source_coverage = call.source_coverage ?? [];
  const context = call.context ?? [];

  if (variant === "empty") {
    const summaryText = t("chat.ledger.summaryEmpty", { dropped: call.dropped_by_gate });
    return (
      <div className="bg-amber/10 border border-amber/30 text-amber px-2 py-0.5 rounded text-xs overflow-x-hidden">
        <div className="flex items-center gap-1 min-w-0">
          <span>⚠</span>
          <span>{summaryText}</span>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="ml-auto bg-transparent border-none p-0 text-amber cursor-pointer text-xs"
            aria-expanded={expanded}
            aria-label={t("chat.ledger.ariaToggle")}
          >
            {expanded ? "▴" : "▾"}
          </button>
        </div>
        {expanded && (
          <div className="mt-1 pt-1 border-t border-amber/30">
            <div className="flex justify-between py-0.5">
              <span className="text-amber/70">{t("chat.ledger.field.query")}</span>
              <span className="font-medium text-amber">{call.query}</span>
            </div>
            <div className="flex justify-between py-0.5">
              <span className="text-amber/70">{t("chat.ledger.field.result")}</span>
              <span className="font-medium text-amber">{t("chat.ledger.summaryEmpty", { dropped: call.dropped_by_gate })}</span>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ----- default -----
  const summaryText = t("chat.ledger.summary", { chunks: context.length, sources: source_coverage.length });
  return (
    <div className="bg-sky/5 border border-border text-blue px-2 py-0.5 rounded text-xs overflow-x-hidden">
      <div className="flex items-center gap-1 min-w-0">
        <span>◉</span>
        <span className="font-mono truncate">{call.query}</span>
        <span className="text-muted">→</span>
        <span>{summaryText}</span>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="ml-auto bg-transparent border-none p-0 text-blue cursor-pointer text-xs"
          aria-expanded={expanded}
          aria-label={t("chat.ledger.ariaToggle")}
        >
          {expanded ? "▴" : "▾"}
        </button>
      </div>

      {expanded && (
        <div className="mt-1 pt-1 border-t border-border">
          {/* Section A — metadata */}
          <div className="flex justify-between py-0.5">
            <span className="text-muted">{t("chat.ledger.field.query")}</span>
            <span className="font-medium text-ink">{call.query}</span>
          </div>
          <div className="flex justify-between py-0.5">
            <span className="text-muted">{t("chat.ledger.field.mode")}</span>
            <span className="font-medium text-ink"><ModeDisplay expected_hits={call.expected_hits} t={t} /></span>
          </div>
          <div className="flex justify-between py-0.5">
            <span className="text-muted">{t("chat.ledger.field.scope")}</span>
            <span className="font-medium text-ink"><ScopeDisplay call={call} t={t} /></span>
          </div>

          {/* Section B — chunk list */}
          {context.length > 0 && (
            <div className="mt-1 pt-1 border-t border-border space-y-0.5">
              {context.map((chunk) => (
                <div key={chunk.chunk_id} className="text-ink min-w-0">
                  <span className="font-mono shrink-0">[{chunk.citation_index}]</span>
                  <span className="truncate shrink">{chunk.source_title}</span>
                  <span className="text-muted"> · </span>
                  <span>{formatTimestamp(chunk.timestamp_start)}–{formatTimestamp(chunk.timestamp_end)}</span>
                  <span className="text-muted"> · </span>
                  <span>{chunk.rerank_score.toFixed(2)}</span>
                  <div className="pl-4 text-muted truncate max-w-xs" title={chunk.preview}>
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
