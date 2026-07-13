import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Minimize2, ArrowLeftToLine, ArrowRightToLine} from 'lucide-react';
import { useLanguage } from "@/app/LanguageContext";
import { api, toErrorMessageWithT } from "@/lib/api";
import type { Artifact, Source, SourceContent } from "@/lib/types";

import { usePanelResize, Resizer, COLLAPSED_PANEL } from "@/components/lists/panel-resize";
import { NavbarTitle } from "@/components/lists/NavbarTitle";
import { LabPanel } from "@/components/lists/LabPanel";
import { ChatPanel } from "@/components/lists/ChatPanel";
import { SourcesViewerMode } from "@/components/lists/sources/SourcesViewerMode";
import { SourcesListMode } from "@/components/lists/sources/SourcesListMode";
import { buildMindmapAskMessage, type OpenSourceOpts, type PendingChatMessage } from "@/lib/chat-utils";

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
  const [targetSection, setTargetSection] = useState<
    { sectionId?: string; timestampStart?: number } | null
  >(null);
  // Carries a keyword-driven message from the digest chip click in
  // SourcesViewerMode (or a mindmap node click in the Lab) to the
  // always-mounted ChatPanel. ChatPanel always acknowledges (via
  // `onPendingMessageConsumed`) so the prop is cleared whether the
  // message is dispatched or rejected. See `PendingChatMessage` in
  // lib/chat-utils for the full shape + rationale.
  const [pendingChatMessage, setPendingChatMessage] = useState<PendingChatMessage | null>(null);
  const tRef = useRef(t);
  tRef.current = t;

  // Reconcile chat scope with the current sources instead of resetting to all:
  // prune deleted ids (a stale id must never reach chat, and deleting the last
  // source clears the selection), keep the user's selection of survivors (so an
  // ingest or a poll doesn't wipe a partial selection), and default genuinely
  // new sources into scope (matching "new content is chattable by default").
  const knownSourceIdsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const currentIds = sources.map((s) => s.id);
    const currentIdSet = new Set(currentIds);
    const known = knownSourceIdsRef.current;
    knownSourceIdsRef.current = currentIdSet;
    setSelectedSourceIds((prev) => {
      const kept = prev.filter((id) => currentIdSet.has(id));
      const added = currentIds.filter((id) => !known.has(id) && !kept.includes(id));
      return added.length === 0 && kept.length === prev.length ? prev : [...kept, ...added];
    });
  }, [sources]);

  const containerRef = useRef<HTMLDivElement>(null);

  const { sourcesW, labW, chatW, onMouseDownLeft, onMouseDownRight } = usePanelResize(
    containerRef,
    sourcesCollapsed,
    labCollapsed,
  );

  const currentSourceIdRef = useRef<string | null>(null);
  const openSourceControllerRef = useRef<AbortController | null>(null);
  const loadControllerRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    // Coordinate concurrent refreshes (mount, delete, ingest-done, viewer
    // close): each load aborts the in-flight one, so an out-of-order response
    // can't clobber fresh state with a stale (or transiently empty) snapshot.
    loadControllerRef.current?.abort();
    const controller = new AbortController();
    loadControllerRef.current = controller;
    const { signal } = controller;
    try {
      const [lists, nextSources, nextArtifacts] = await Promise.all([
        api.listLists({ signal }),
        api.listSources(listId, { signal }),
        api.listArtifacts(listId, { signal }),
      ]);
      const current = lists?.find((l) => l.id === listId);
      setListName(current?.name ?? tRef.current("lists.listWorkspace"));
      setSources(nextSources ?? []);
      setArtifacts(nextArtifacts ?? []);
      setLoadError(null);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      setLoadError(toErrorMessageWithT(err, tRef.current));
    }
  }, [listId]);

  useEffect(() => {
    void load();
    return () => loadControllerRef.current?.abort();
  }, [load]);

  const prevDetailSourceRef = useRef<Source | null>(null);
  useEffect(() => {
    if (prevDetailSourceRef.current !== null && detailSource === null) {
      void load();
    }
    prevDetailSourceRef.current = detailSource;
  }, [detailSource, load]);

  useEffect(() => {
    return () => {
      if (openSourceControllerRef.current) {
        openSourceControllerRef.current.abort();
      }
    };
  }, []);

  function handleOpenSource(
    source: Source,
    opts?: OpenSourceOpts,
  ) {
    currentSourceIdRef.current = source.id;
    setDetailSource(source);
    setSourceContent(null);
    setTargetSection(
      opts?.sectionId || opts?.timestampStart != null
        ? { sectionId: opts.sectionId, timestampStart: opts.timestampStart }
        : null,
    );
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

  // Buffer a chat message from the digest chip; ChatPanel picks it up
  // via prop and acknowledges (sends or rejects) it. The page does
  // not gate on chat state — chat owns the decision.
  function handleDiscussKeyword(message: string) {
    setPendingChatMessage({ text: message, nonce: Date.now() });
  }

  // Buffer a chat message from a mindmap node click. The topic label
  // and (optional) parent label are composed into a localized
  // "Discuss X" / "Discuss X, in the larger context of Y" string here
  // so the mindmap component only emits structured data. The chat
  // scope is locked to the artifact's persistent source_ids so the
  // generated mindmap stays coherent with the question.
  function handleAskInChatFromMindmap(
    topic: string,
    parentTopic: string | null,
    sourceIds: string[],
    evidence: string,
  ) {
    const text = buildMindmapAskMessage(t, topic, parentTopic, evidence);
    setPendingChatMessage({ text, nonce: Date.now(), sourceIds });
  }

  async function handleSaveToArtifact(messageId: string) {
    try {
      await api.saveChatMessage(listId, messageId);
      const next = await api.listArtifacts(listId);
      setArtifacts(next ?? []);
    } catch (err) {
      setLoadError(toErrorMessageWithT(err, tRef.current));
    }
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
              onClick={
                detailSource
                  ? () => {
                      setDetailSource(null);
                      // Clear the citation jump target so the next open
                      // (e.g. via the sources list) lands on section 0
                      // instead of the previous click's cited section.
                      setTargetSection(null);
                    }
                  : () => setSourcesCollapsed((v) => !v)
              }
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
                  listId={listId}
                  targetSection={targetSection}
                  onRefresh={() => {
                    if (detailSource) {
                      void api.getSource(detailSource.id).then((content) => {
                        if (currentSourceIdRef.current !== detailSource.id) return;
                        setSourceContent(content ?? null);
                      });
                    }
                  }}
                  onDiscussKeyword={handleDiscussKeyword}
                />
              ) : (
                <SourcesListMode
                  listId={listId}
                  sources={sources}
                  selectedSourceIds={selectedSourceIds}
                  onSelectedSourcesChange={setSelectedSourceIds}
                  onRefresh={load}
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
          <ChatPanel
            selectedSourceIds={selectedSourceIds}
            sources={sources}
            listId={listId}
            onOpenSource={handleOpenSource}
            pendingMessage={pendingChatMessage}
            onPendingMessageConsumed={() => setPendingChatMessage(null)}
            onSaveToArtifact={handleSaveToArtifact}
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
            sources={sources}
            onArtifactsChange={setArtifacts}
            onToggleCollapse={() => setLabCollapsed((v) => !v)}
            onAskInChatFromMindmap={handleAskInChatFromMindmap}
            onOpenSource={handleOpenSource}
          />
        </div>
      </div>
    </>
  );
}
