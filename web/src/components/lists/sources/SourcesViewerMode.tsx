import { MdClose } from "react-icons/md";

import { useLanguage } from "@/app/LanguageContext";
import type { Source, SourceContent } from "@/lib/types";
import { Banner } from "@/components/lists/Banner";
import { DigestAccordion } from "@/components/lists/DigestAccordion";

export function SourcesViewerMode({
  source,
  sourceContent,
  onClose,
}: {
  source: Source;
  sourceContent: SourceContent | null;
  onClose: () => void;
}) {
  const { t } = useLanguage();
  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-start gap-3 border-b border-border px-4 py-4">
        <button
          type="button"
          onClick={onClose}
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
          aria-label={t("lists.closeViewer")}
        >
          <MdClose size={16} />
        </button>
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-sm font-medium text-ink">{source.title}</p>
          <p className="m-0 mt-0.5 text-xs text-muted">{source.platform}</p>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {sourceContent && (
          <Banner
            sourceId={source.id}
            sourceUrl={sourceContent.source_url}
            uploader={sourceContent.uploader}
            durationSeconds={sourceContent.duration_seconds}
          />
        )}
        {sourceContent && (
          <DigestAccordion
            summary={sourceContent.summary}
            keywords={sourceContent.keywords}
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
