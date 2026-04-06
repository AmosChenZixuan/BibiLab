import { useState } from "react";
import { MdExpandLess, MdExpandMore } from "react-icons/md";

import { useLanguage } from "@/app/LanguageContext";

export function DigestAccordion({ summary, keywords }: { summary: string; keywords: string[] }) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(true);
  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-blue/25">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between gap-3 border-0 bg-transparent px-4 py-3.5 text-left transition hover:bg-sky/6"
      >
        <span className="text-sm font-semibold text-ink">{t("lists.digest")}</span>
        {expanded ? <MdExpandLess size={18} className="shrink-0 text-muted" /> : <MdExpandMore size={18} className="shrink-0 text-muted" />}
      </button>
      {expanded && (
        <div className="border-t border-border px-4 py-4 space-y-3">
          {summary && (
            <p className="text-sm text-muted leading-relaxed m-0">{summary}</p>
          )}
          {keywords.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {keywords.map((kw) => (
                <span
                  key={kw}
                  className="inline-block rounded-full bg-blue/10 px-2.5 py-0.5 text-xs text-blue/80"
                >
                  {kw}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
