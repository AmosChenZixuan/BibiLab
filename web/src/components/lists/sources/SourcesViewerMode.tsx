
import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import type { Source, SourceContent } from "@/lib/types";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { Banner } from "@/components/lists/Banner";
import { DigestAccordion } from "@/components/lists/DigestAccordion";

export function SourcesViewerMode({
  source,
  sourceContent,
  onRefresh,
  listId,
}: {
  source: Source;
  sourceContent: SourceContent | null;
  onRefresh: () => void;
  listId: string;
}) {
  const { t } = useLanguage();
  const { trackJobs } = useJobActivity();

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
            source={source}
            summary={sourceContent.summary}
            keywords={sourceContent.keywords}
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
