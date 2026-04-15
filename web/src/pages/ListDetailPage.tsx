import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Minimize2, ArrowLeftToLine, ArrowRightToLine} from 'lucide-react';
import { useLanguage } from "@/app/LanguageContext";
import { api, toErrorMessageWithT } from "@/lib/api";
import { ARTIFACT_TYPE_KEYS } from "@/lib/artifactTypes";
import type { Artifact, Source, SourceContent } from "@/lib/types";

import { usePanelResize, Resizer, COLLAPSED_PANEL } from "@/components/lists/panel-resize";
import { NavbarTitle } from "@/components/lists/NavbarTitle";
import { LabPanel } from "@/components/lists/LabPanel";
import { SourcesViewerMode } from "@/components/lists/sources/SourcesViewerMode";
import { SourcesListMode } from "@/components/lists/sources/SourcesListMode";

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
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const [labCollapsed, setLabCollapsed] = useState(false);
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);

  useEffect(() => {
    if (sources.length > 0) {
      setSelectedSourceIds(sources.map((s) => s.id));
    }
  }, [sources]);

  const containerRef = useRef<HTMLDivElement>(null);

  const { sourcesW, labW, chatW, onMouseDownLeft, onMouseDownRight } = usePanelResize(
    containerRef,
    sourcesCollapsed,
    labCollapsed,
  );

  const currentSourceIdRef = useRef<string | null>(null);
  const openSourceControllerRef = useRef<AbortController | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [lists, nextSources, nextArtifacts] = await Promise.all([
        api.listLists({ signal }),
        api.listSources(listId, { signal }),
        api.listArtifacts(listId, { signal }),
      ]);
      const current = lists?.find((l) => l.id === listId);
      setListName(current?.name ?? t("lists.listWorkspace"));
      setSources(nextSources ?? []);
      setArtifacts(nextArtifacts ?? []);
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

  useEffect(() => {
    return () => {
      if (openSourceControllerRef.current) {
        openSourceControllerRef.current.abort();
      }
    };
  }, []);

  function handleOpenSource(source: Source) {
    currentSourceIdRef.current = source.id;
    setDetailSource(source);
    setSourceContent(null);
    if (openSourceControllerRef.current) {
      openSourceControllerRef.current.abort();
    }
    const controller = new AbortController();
    openSourceControllerRef.current = controller;
    void api.getSource(source.id, { signal: controller.signal }).then((content) => {
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

  const handleArtifactGenerated = useCallback((artifactId: string, type: Artifact["type"], sourceIds: string[]) => {
    const placeholder: Artifact = {
      id: artifactId,
      name: t(ARTIFACT_TYPE_KEYS[type] ?? "lab.reportsModal.custom"),
      type,
      prompt: "",
      source_ids: sourceIds,
      status: "generating",
      created_at: new Date().toISOString(),
    };
    setArtifacts((prev) => [placeholder, ...prev]);
  }, [t]);

  const panelBase = "flex h-full shrink-0 flex-col overflow-hidden rounded-3xl border border-border bg-white/76 shadow-lg";

  return (
    <>
      <NavbarTitle name={listName} onCommit={handleRenameCommit} />

      <div
        ref={containerRef}
        className="fixed inset-x-0 top-14 bottom-0 z-0 box-border flex overflow-hidden px-4 pb-4"
      >
        {/* ── Sources panel ── */}
        <div
          style={
            sourcesCollapsed
              ? { width: `${COLLAPSED_PANEL}px`, minWidth: `${COLLAPSED_PANEL}px` }
              : { width: `${sourcesW}px` }
          }
          className={panelBase}
        >
          <div className="flex shrink-0 items-center border-b border-border px-4 py-4">
            {!sourcesCollapsed && (
              <h2 className="m-0 flex-1 font-serif text-lg text-ink">{t("lists.sources")}</h2>
            )}
            <button
              type="button"
              onClick={detailSource ? () => setDetailSource(null) : () => setSourcesCollapsed((v) => !v)}
              aria-label={detailSource ? "Close viewer" : sourcesCollapsed ? "Expand sources" : "Collapse sources"}
              className={`flex h-7 w-7 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink ${sourcesCollapsed ? "mx-auto" : ""}`}
            >
              {detailSource ? <Minimize2 size={16} /> : (sourcesCollapsed ? <ArrowRightToLine size={16} /> : <ArrowLeftToLine size={16} />)}
            </button>
          </div>

          {!sourcesCollapsed && (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              {loadError ? (
                <p className="m-0 px-4 py-3 text-sm text-pink">{loadError}</p>
              ) : detailSource ? (
                <SourcesViewerMode
                  source={detailSource}
                  sourceContent={sourceContent}
                  onRefresh={() => {
                    if (detailSource) {
                      void api.getSource(detailSource.id).then((content) => {
                        if (currentSourceIdRef.current !== detailSource.id) return;
                        setSourceContent(content ?? null);
                      });
                    }
                  }}
                />
              ) : (
                <SourcesListMode
                  listId={listId}
                  sources={sources}
                  selectedSourceIds={selectedSourceIds}
                  onSelectedSourcesChange={setSelectedSourceIds}
                  onOpenSource={handleOpenSource}
                />
              )}
            </div>
          )}
        </div>

        <Resizer onMouseDown={onMouseDownLeft} />

        {/* ── Chat panel ── */}
        <div
          style={{ width: `${chatW}px` }}
          className={panelBase}
        >
          <SkeletonPanel
            title="Chat"
            note="list-scoped chat arrives in v1"
          />
        </div>

        <Resizer onMouseDown={onMouseDownRight} />

        {/* ── Lab panel ── */}
        <div
          style={
            labCollapsed
              ? { width: `${COLLAPSED_PANEL}px`, minWidth: `${COLLAPSED_PANEL}px`, marginLeft: "auto" }
              : { width: `${labW}px` }
          }
          className={panelBase}
        >
          <LabPanel
            listId={listId}
            labCollapsed={labCollapsed}
            labW={labW}
            selectedSourceIds={selectedSourceIds}
            artifacts={artifacts}
            onArtifactsChange={setArtifacts}
            onToggleCollapse={() => setLabCollapsed((v) => !v)}
            onArtifactGenerated={handleArtifactGenerated}
          />
        </div>
      </div>
    </>
  );
}
