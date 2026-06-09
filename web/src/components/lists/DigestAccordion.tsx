import { useState } from "react";
import { ChevronUp, ChevronDown, MoreVertical, RotateCcw, Pencil } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { ContextMenu } from "@/components/ui/ContextMenu";
import { DigestFacets, type Facets } from "@/components/lists/DigestFacets";
import { PagerTabs } from "@/components/lists/PagerTabs";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { useDismissOnDone } from "@/components/jobs/useDismissOnDone";
import type { DigestJob, SourceFacetsPatch, SourceSection } from "@/lib/types";

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
  onRefresh,
  facets,
  onSaveFacets,
  listId,
  sections,
}: {
  source: { id: string };
  summary: string;
  keywords: string[];
  onRerun: (sourceId: string) => void;
  onRefresh: (sourceId: string) => void;
  facets: Facets;
  onSaveFacets: (patch: SourceFacetsPatch) => Promise<void>;
  listId: string;
  sections?: SourceSection[];
}) {
  const { t } = useLanguage();
  const [expanded, setExpanded] = useState(true);
  const [editingFacets, setEditingFacets] = useState(false);
  const [activeSectionIdx, setActiveSectionIdx] = useState(0);
  const { getJobs } = useJobActivity();

  const digestJobs = getJobs("digest" as const, listId).filter(
    (item) => (item.job as DigestJob).meta?.source_id === source.id,
  );

  // When a digest job reaches done, refetch the source and dismiss.
  useDismissOnDone({
    jobs: digestJobs,
    onDone: () => onRefresh(source.id),
  });

  let activeJob: typeof digestJobs[0] | undefined;
  for (const item of digestJobs) {
    if (!item.isTerminal) {
      if (!activeJob) activeJob = item;
    }
  }

  const handleRerun = async () => {
    await onRerun(source.id);
  };

  // 1-section case: byte-identical to today. No pager, no per-section block.
  // The PagerTabs path is gated on `sections && sections.length > 1` so the
  // 1-section markup is unchanged (regression guard).
  const showPager = Array.isArray(sections) && sections.length > 1;
  const activeSection = showPager ? sections[activeSectionIdx] : null;
  const visibleSummary = activeSection ? activeSection.summary : summary;
  const visibleKeywords = activeSection ? activeSection.keywords : keywords;

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-blue/25">
      {/* Header */}
      <div className="flex w-full items-center justify-between gap-3 border-0 bg-transparent px-4 py-3.5">
        <span className="text-sm font-semibold text-ink">{t("lists.digest")}</span>

        {/* Right side: context menu / loading, then expand arrow */}
        <div className="flex items-center gap-1">
          {activeJob ? (
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
          {showPager && (
            // Negative-margin wrapper so the pager's hairline border runs
            // edge-to-edge, matching the outer `border-t` above. The 1-section
            // path doesn't render this wrapper at all, so its DOM is
            // byte-identical to the pre-sections markup.
            <div className="-mx-4">
              <PagerTabs
                sections={sections}
                activeIdx={activeSectionIdx}
                onActiveIdxChange={setActiveSectionIdx}
              />
            </div>
          )}
          <div className={`transition-opacity duration-300 ${activeJob ? "opacity-30" : "opacity-100"}`}>
            {visibleSummary ? (
              <p className="m-0 text-sm leading-relaxed text-muted">{visibleSummary}</p>
            ) : (
              <div className="space-y-1.5">
                <div className="h-3 w-3/4 rounded bg-blue/20" />
                <div className="h-3 w-1/2 rounded bg-blue/20" />
              </div>
            )}
          </div>

          <div className={`transition-opacity duration-300 ${activeJob ? "opacity-30" : "opacity-100"}`}>
            {visibleKeywords.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {visibleKeywords.map((kw) => (
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
