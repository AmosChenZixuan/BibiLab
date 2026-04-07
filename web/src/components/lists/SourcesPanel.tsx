import { MdChevronLeft, MdChevronRight } from "react-icons/md";

import type { Source, SourceContent } from "@/lib/types";
import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import { SourcesViewerMode } from "@/components/lists/sources/SourcesViewerMode";
import { SourcesListMode } from "@/components/lists/sources/SourcesListMode";
import { COLLAPSED_PANEL, MIN_PANEL } from "@/components/lists/panel-resize";

interface SourcesPanelProps {
  listId: string;
  sources: Source[];
  detailSource: Source | null;
  sourceContent: SourceContent | null;
  loadError: string | null;
  sourcesCollapsed: boolean;
  sourcesW: number;
  onToggleCollapse: () => void;
  onOpenSource: (source: Source) => void;
  onCloseSource: () => void;
  onRefreshSource: () => void;
  currentSourceIdRef: React.MutableRefObject<string | null>;
}

export function SourcesPanel({
  listId,
  sources,
  detailSource,
  sourceContent,
  loadError,
  sourcesCollapsed,
  sourcesW,
  onToggleCollapse,
  onOpenSource,
  onCloseSource,
  onRefreshSource,
  currentSourceIdRef,
}: SourcesPanelProps) {
  const { t } = useLanguage();
  const panelBase = "flex shrink-0 flex-col overflow-hidden rounded-3xl border border-border bg-white/76 shadow-lg";

  function handleRefreshSource() {
    if (detailSource) {
      void api.getSource(detailSource.id).then(() => {
        if (currentSourceIdRef.current !== detailSource.id) return;
        onRefreshSource();
      });
    }
  }

  return (
    <div
      style={
        sourcesCollapsed
          ? { width: `${COLLAPSED_PANEL}px`, minWidth: `${COLLAPSED_PANEL}px` }
          : { width: `${sourcesW}px`, minWidth: `${MIN_PANEL}px` }
      }
      className={panelBase}
    >
      <div className="flex shrink-0 items-center border-b border-border px-4 py-4">
        {!sourcesCollapsed && (
          <h2 className="m-0 flex-1 font-serif text-lg text-ink">{t("lists.sources")}</h2>
        )}
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label={sourcesCollapsed ? "expand sources" : "collapse sources"}
          className={`flex h-7 w-7 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink ${sourcesCollapsed ? "mx-auto" : ""}`}
        >
          {sourcesCollapsed ? <MdChevronRight size={16} /> : <MdChevronLeft size={16} />}
        </button>
      </div>

      {!sourcesCollapsed && (
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {loadError ? (
            <p className="m-0 px-4 py-3 text-sm text-rose-900">{loadError}</p>
          ) : detailSource ? (
            <SourcesViewerMode
              source={detailSource}
              sourceContent={sourceContent}
              onClose={onCloseSource}
              onRefresh={handleRefreshSource}
            />
          ) : (
            <SourcesListMode
              listId={listId}
              sources={sources}
              onOpenSource={onOpenSource}
            />
          )}
        </div>
      )}
    </div>
  );
}
