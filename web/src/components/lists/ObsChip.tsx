import { useState } from "react";
import { useLanguage } from "@/app/LanguageContext";

import { Info, Loader2 } from "lucide-react";

import type { RagCall, SearchMode } from "@/lib/chat-utils";

interface ObsChipProps {
  call: RagCall;
}

export function ObsChip({ call }: ObsChipProps) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(false);

  const label = t("chat.obsChip.chipLabel", {
    count: call.candidates_evaluated,
    hits: call.sources_with_hits,
    total: call.sources_total,
  });

  return (
    <div className="relative inline-flex items-center gap-1 px-2 py-0.5 rounded bg-sky/10 border border-border text-xs text-blue">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1 bg-transparent border-none p-0 text-blue cursor-pointer"
        aria-expanded={expanded}
        aria-label={t("chat.obsChip.ariaLabel")}
      >
        <Info size={11} />
        <span className="font-mono">{call.query}</span>
        <span className="text-muted">·</span>
        <span>{label}</span>
      </button>

      {expanded && (
        <div className="absolute top-full left-0 z-float mt-1 bg-white border border-border rounded-lg shadow-lg p-3 min-w-64 text-xs">
          <div className="flex justify-between items-center py-0.5">
            <span className="text-muted">{t("chat.obsChip.query")}</span>
            <span className="font-medium text-ink">{call.query}</span>
          </div>
          <div className="flex justify-between items-center py-0.5">
            <span className="text-muted">{t("chat.obsChip.mode")}</span>
            <span className="font-medium text-ink">{call.search_mode}</span>
          </div>
          <div className="flex justify-between items-center py-0.5">
            <span className="text-muted">{t("chat.obsChip.chunksEvaluated")}</span>
            <span className="font-medium text-ink">{call.candidates_evaluated}</span>
          </div>
          <div className="flex justify-between items-center py-0.5">
            <span className="text-muted">{t("chat.obsChip.sourcesRetrieved")}</span>
            <span className="font-medium text-ink">{call.sources_with_hits} / {call.sources_total}</span>
          </div>
          {call.source_coverage.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border">
              {call.source_coverage.map((s, i) => (
                <span key={i} className="block text-ink">{s.title}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function PendingObsChip({ query, search_mode }: { query: string; search_mode: SearchMode }) {
  return (
    <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-sky/10 border border-border text-xs text-blue">
      <Loader2 size={11} className="animate-spin" />
      <span className="font-mono">{query}</span>
      <span className="text-muted">·</span>
      <span>{search_mode}</span>
    </div>
  );
}

export function PendingMetaChip({ query_type }: { query_type: string }) {
  const { t } = useLanguage();
  const key = `chat.obsChip.metaQueryType.${query_type}`;
  const translated = t(key);
  const label = translated !== key ? translated : query_type;
  return (
    <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-sky/10 border border-border text-xs text-blue">
      <Loader2 size={11} className="animate-spin" />
      <span className="font-mono">{label}</span>
    </div>
  );
}
