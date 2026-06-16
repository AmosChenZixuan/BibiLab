
import { useEffect, useState } from "react";
import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import type { Source, SourceContent, SourceSection } from "@/lib/types";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { Banner } from "@/components/lists/Banner";
import { DigestAccordion } from "@/components/lists/DigestAccordion";

/** Resolve a citation-jump target to a section index. The `sectionId`
 *  branch is the precise match (the chat citation carries the section
 *  id directly so it can land on the cited section even when the
 *  chunk-anchored timestamp falls at a boundary or outside the range);
 *  `timestampStart` stays as a fallback for legacy citations. Returns
 *  0 when no target is supplied or the target doesn't match any
 *  section. */
function resolveTargetIdx(
  sections: SourceSection[] | undefined,
  target?: { sectionId?: string; timestampStart?: number } | null,
): number {
  if (!target || !sections || sections.length === 0) return 0;
  if (target.sectionId) {
    const i = sections.findIndex((s) => s.section_id === target.sectionId);
    if (i >= 0) return i;
  }
  if (target.timestampStart != null) {
    const ts = target.timestampStart;
    const i = sections.findIndex(
      (s) => ts >= s.timestamp_start && ts <= s.timestamp_end,
    );
    if (i >= 0) return i;
  }
  return 0;
}

export function SourcesViewerMode({
  source,
  sourceContent,
  onRefresh,
  listId,
  targetSection,
  onDiscussKeyword,
}: {
  source: Source;
  sourceContent: SourceContent | null;
  onRefresh: () => void;
  listId: string;
  targetSection?: { sectionId?: string; timestampStart?: number } | null;
  onDiscussKeyword?: (message: string) => void;
}) {
  const { t } = useLanguage();
  const { trackJobs } = useJobActivity();
  const [sections, setSections] = useState<SourceSection[] | undefined>(undefined);

  // Fetch sections alongside the source content so the digest body can
  // switch to the sectioned layout for N>1. Failures degrade silently —
  // DigestAccordion falls through to the 1-section path when `sections`
  // is undefined or empty. The `cancelled` flag (per web/CLAUDE.md) is
  // enough to guard stale responses: the `key={source.id}` on
  // <DigestAccordion> remounts the body on source switch, so the new
  // source's state always starts fresh.
  useEffect(() => {
    if (!sourceContent) {
      setSections(undefined);
      return;
    }
    setSections(undefined);
    let cancelled = false;
    void api
      .getSourceSections(source.id)
      .then((rows) => {
        if (cancelled) return;
        setSections(rows ?? []);
      })
      .catch(() => {
        if (cancelled) return;
        // Silent degrade: the digest renders the 1-section path.
        setSections([]);
      });
    return () => {
      cancelled = true;
    };
  }, [source.id, sourceContent]);

  const handleRerunDigest = async (sourceId: string) => {
    try {
      const result = await api.rerunDigest(sourceId);
      if (result?.job_id) {
        trackJobs([{ id: result.job_id, producer: "digest", label: source.title, contextKey: listId }]);
        onRefresh();
      }
    } catch (err) {
      console.error("rerunDigest failed:", err);
    }
  };
  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-start px-4 py-4">
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-lg font-medium text-ink">{source.title}</p>
          <p className="m-0 mt-0.5 text-xs text-muted">{source.platform}</p>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {sourceContent && (
          <Banner
            source={source}
            sourceUrl={sourceContent.source_url}
            uploader={sourceContent.uploader}
            durationSeconds={sourceContent.duration_seconds}
          />
        )}
        {sourceContent && (
          <DigestAccordion
            key={source.id}
            source={source}
            onRerun={handleRerunDigest}
            onRefresh={() => onRefresh()}
            facets={{
              seriesName: sourceContent.series_name,
              sequenceNumber: sourceContent.sequence_number,
              seasonNumber: sourceContent.season_number,
            }}
            onSaveFacets={async (patch) => {
              await api.updateSourceFacets(source.id, patch);
              onRefresh();
            }}
            listId={listId}
            sections={sections}
            initialActiveIdx={resolveTargetIdx(sections, targetSection)}
            onDiscussKeyword={onDiscussKeyword}
          />
        )}
        {sourceContent?.transcript && (
          <>
            <p className="text-xs font-semibold uppercase tracking-wider text-muted/70">{t("lists.transcript")}</p>
            <pre className="p-1 m-0 whitespace-pre-wrap font-mono text-xs text-muted leading-relaxed">
              {sourceContent.transcript}
            </pre>
          </>
        )}
      </div>
    </div>
  );
}
