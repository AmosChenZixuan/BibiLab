import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import { useLanguage } from "@/app/LanguageContext";
import { api, toErrorMessageWithT } from "@/lib/api";
import type { Source, SourceContent } from "@/lib/types";

import { usePanelResize, Resizer, MIN_PANEL } from "@/components/lists/panel-resize";
import { NavbarTitle } from "@/components/lists/NavbarTitle";
import { SourcesPanel } from "@/components/lists/SourcesPanel";

function SkeletonPanel({ title, note }: { title: string; note: string }) {
  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-border px-5 py-4">
        <h2 className="m-0 font-serif text-lg text-ink">{title}</h2>
      </div>
      <div className="flex flex-1 flex-col items-center justify-center gap-4 px-6 py-8">
        <div className="w-full space-y-2.5">
          <div className="h-2.5 w-5/6 rounded-full bg-linear-to-r from-pink/12 to-sky/12" />
          <div className="h-2.5 rounded-full bg-linear-to-r from-pink/12 to-sky/12" />
          <div className="h-2.5 w-2/3 rounded-full bg-linear-to-r from-pink/12 to-sky/12" />
        </div>
        <p className="m-0 text-center text-sm text-muted/80">{note}</p>
      </div>
    </div>
  );
}

export function ListDetailPage() {
  const { t } = useLanguage();
  const { listId = "" } = useParams();
  const [listName, setListName] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [detailSource, setDetailSource] = useState<Source | null>(null);
  const [sourceContent, setSourceContent] = useState<SourceContent | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const { sourcesW, labW, chatW, onMouseDownLeft, onMouseDownRight } = usePanelResize(
    containerRef,
    sourcesCollapsed,
  );

  const currentSourceIdRef = useRef<string | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [lists, nextSources] = await Promise.all([
        api.listLists({ signal }),
        api.listSources(listId, { signal }),
      ]);
      const current = lists?.find((l) => l.id === listId);
      setListName(current?.name ?? t("lists.listWorkspace"));
      setSources(nextSources ?? []);
      setLoadError(null);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      setLoadError(toErrorMessageWithT(err, t));
    }
  }, [listId, t]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  function handleOpenSource(source: Source) {
    currentSourceIdRef.current = source.id;
    setDetailSource(source);
    setSourceContent(null);
    void api.getSource(source.id).then((content) => {
      if (currentSourceIdRef.current !== source.id) return;
      setSourceContent(content ?? null);
    }).catch(() => {
      setSourceContent(null);
    });
  }

  async function handleRenameCommit(newName: string) {
    try {
      const updated = await api.updateList(listId, { name: newName });
      if (!updated) return;
      setListName(updated.name);
    } catch {
      // On failure the portal reverts its own draft via the name prop
    }
  }

  const panelBase = "flex shrink-0 flex-col overflow-hidden rounded-3xl border border-border bg-white/76 shadow-lg";

  return (
    <>
      <NavbarTitle name={listName} onCommit={handleRenameCommit} />

      <div
        ref={containerRef}
        className="fixed inset-x-0 top-14 bottom-0 z-0 box-border flex overflow-hidden px-4 pb-4"
      >
        <SourcesPanel
          listId={listId}
          sources={sources}
          detailSource={detailSource}
          sourceContent={sourceContent}
          loadError={loadError}
          sourcesCollapsed={sourcesCollapsed}
          sourcesW={sourcesW}
          onToggleCollapse={() => setSourcesCollapsed((v) => !v)}
          onOpenSource={handleOpenSource}
          onCloseSource={() => setDetailSource(null)}
          onRefreshSource={() => setSourceContent((prev) => prev)}
          currentSourceIdRef={currentSourceIdRef}
        />

        <Resizer onMouseDown={onMouseDownLeft} />

        {/* ── Chat panel ── */}
        <div
          style={{ width: `${chatW}px`, minWidth: `${MIN_PANEL}px` }}
          className={panelBase}
        >
          <SkeletonPanel
            title={t("lists.chat")}
            note={t("lists.chatDesc")}
          />
        </div>

        <Resizer onMouseDown={onMouseDownRight} />

        {/* ── Lab panel ── */}
        <div
          style={{ width: `${labW}px`, minWidth: `${MIN_PANEL}px` }}
          className={panelBase}
        >
          <SkeletonPanel
            title={t("lists.lab")}
            note={t("lists.labDesc")}
          />
        </div>
      </div>
    </>
  );
}
