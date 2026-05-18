import { useState } from "react";
import { ChevronUp, ChevronDown, MoreVertical, RotateCcw, Pencil } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { ContextMenu } from "@/components/ui/ContextMenu";
import { DigestFacets, type Facets } from "@/components/lists/DigestFacets";
import type { SourceFacetsPatch } from "@/lib/types";

function LoadingDots() {
  return (
    <span className="flex items-center gap-0.5">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="block h-1 w-1 rounded-full bg-current digest-loading-dot"
          style={{
            animationDelay: `${i * 0.15}s`,
          }}
        />
      ))}
    </span>
  );
}

export function DigestAccordion({
  source,
  summary,
  keywords,
  onRerun,
  facets,
  onSaveFacets,
}: {
  source: { id: string };
  summary: string;
  keywords: string[];
  onRerun: (sourceId: string) => void;
  facets: Facets;
  onSaveFacets: (patch: SourceFacetsPatch) => Promise<void>;
}) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(true);
  const [loading, setLoading] = useState(false);
  const [editingFacets, setEditingFacets] = useState(false);

  const handleRerun = async () => {
    setLoading(true);
    try {
      await onRerun(source.id);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-blue/25">
      {/* Header */}
      <div className="flex w-full items-center justify-between gap-3 border-0 bg-transparent px-4 py-3.5">
        <span className="text-sm font-semibold text-ink">{t("lists.digest")}</span>

        {/* Right side: context menu / loading, then expand arrow */}
        <div className="flex items-center gap-1">
          {loading ? (
            <LoadingDots />
          ) : (
            <ContextMenu
              items={[
                {
                  label: t("lists.facets.edit"),
                  icon: <Pencil size={14} />,
                  onClick: () => {
                    setExpanded(true);
                    setEditingFacets(true);
                  },
                },
                {
                  label: t("lists.rerunDigest"),
                  icon: <RotateCcw size={14} />,
                  onClick: handleRerun,
                },
              ]}
              trigger={({ toggle, triggerRef }) => (
                <button
                  ref={triggerRef}
                  type="button"
                  aria-label="Digest options"
                  onClick={toggle}
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
                >
                  <MoreVertical size={16} />
                </button>
              )}
            />
          )}
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-label={expanded ? "Collapse" : "Expand"}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
          >
            {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
        </div>
      </div>

      {/* Content */}
      {expanded && (
        <div className="border-t border-border px-4 py-4 space-y-3">
          <DigestFacets
            key={editingFacets ? "facets-edit" : "facets-read"}
            facets={facets}
            editing={editingFacets}
            onSave={onSaveFacets}
            onExitEdit={() => setEditingFacets(false)}
          />
          <div className={`transition-opacity duration-300 ${loading ? "opacity-30" : "opacity-100"}`}>
            {summary ? (
              <p className="m-0 text-sm leading-relaxed text-muted">{summary}</p>
            ) : (
              <div className="space-y-1.5">
                <div className="h-3 w-3/4 rounded bg-blue/20" />
                <div className="h-3 w-1/2 rounded bg-blue/20" />
              </div>
            )}
          </div>

          <div className={`transition-opacity duration-300 ${loading ? "opacity-30" : "opacity-100"}`}>
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
        </div>
      )}
    </div>
  );
}
