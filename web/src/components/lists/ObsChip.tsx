import { useState } from "react";
import { useLanguage } from "@/app/LanguageContext";

import { Info } from "lucide-react";

import type { RagMetadata } from "@/lib/chat-utils";

interface ObsChipProps {
  rag: RagMetadata;
}

export function ObsChip({ rag }: ObsChipProps) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(false);

  const label = t("chat.obsChip.chipLabel", {
    count: rag.candidates_evaluated,
    hits: rag.sources_with_hits,
    total: rag.sources_total,
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
        <span>{label}</span>
      </button>

      {expanded && (
        <div className="absolute top-full left-0 z-float mt-1 bg-white border border-border rounded-lg shadow-lg p-3 min-w-64 text-xs">
          <div className="flex justify-between items-center py-0.5">
            <span className="text-muted">{t("chat.obsChip.mode")}</span>
            <span className="font-medium text-ink">{rag.mode}</span>
          </div>
          <div className="flex justify-between items-center py-0.5">
            <span className="text-muted">{t("chat.obsChip.chunksEvaluated")}</span>
            <span className="font-medium text-ink">{rag.candidates_evaluated}</span>
          </div>
          <div className="flex justify-between items-center py-0.5">
            <span className="text-muted">{t("chat.obsChip.sourcesRetrieved")}</span>
            <span className="font-medium text-ink">{rag.sources_with_hits} / {rag.sources_total}</span>
          </div>
          {rag.sources.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border">
              {rag.sources.map((s, i) => (
                <span key={i} className="block text-ink">{s.title}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
