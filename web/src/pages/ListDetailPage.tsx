import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  MdChevronLeft,
  MdChevronRight,
} from "react-icons/md";

import { api, toErrorMessage } from "@/lib/api";
import type { NoteContent, Source } from "@/lib/types";

import { usePanelResize, Resizer, MIN_PANEL, COLLAPSED_PANEL } from "@/components/lists/panel-resize";
import { NavbarTitle } from "@/components/lists/NavbarTitle";
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
  const { listId = "" } = useParams();
  const [listName, setListName] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [detailSource, setDetailSource] = useState<Source | null>(null);
  const [note, setNote] = useState<NoteContent | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const { sourcesW, labW, chatW, onMouseDownLeft, onMouseDownRight } = usePanelResize(
    containerRef,
    sourcesCollapsed,
  );

  const currentSourceIdRef = useRef<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [lists, nextSources] = await Promise.all([api.listLists(), api.listSources(listId)]);
      const current = lists.find((l) => l.id === listId);
      setListName(current?.name ?? "List workspace");
      setSources(nextSources);
      setLoadError(null);
    } catch (err) {
      setLoadError(toErrorMessage(err));
    }
  }, [listId]);

  useEffect(() => {
    void load();
  }, [load]);

  function handleOpenSource(source: Source) {
    currentSourceIdRef.current = source.video_id;
    setDetailSource(source);
    setNote(null);
    setTranscript(null);
    setTranscriptError(null);
    void api.getNoteContent(source.video_id).then((note) => {
      if (currentSourceIdRef.current !== source.video_id) return;
      const rewritten = note.markdown.replace(
        /!\[([^\]]*)\]\(([^)]+)\)/g,
        (_match: string, alt: string, src: string) => {
          if (src.startsWith("http://") || src.startsWith("https://")) return _match;
          const file = src.replace(/^attachments\//, "");
          return `![${alt}](/api/notes/${source.video_id}/attachments/${file})`;
        },
      );
      setNote({ ...note, markdown: rewritten });
    }).catch(() => setNote({ video_id: source.video_id, title: source.title, markdown: "" }));
    void api.getNoteTranscript(source.video_id)
      .then((res) => {
        if (currentSourceIdRef.current !== source.video_id) return;
        setTranscript(res.text);
      })
      .catch(() => { setTranscriptError("Failed to load transcript"); });
  }

  async function handleRenameCommit(newName: string) {
    try {
      const updated = await api.updateList(listId, { name: newName });
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
        {/* ── Sources panel ── */}
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
              <h2 className="m-0 flex-1 font-serif text-lg text-ink">Sources</h2>
            )}
            <button
              type="button"
              onClick={() => setSourcesCollapsed((v) => !v)}
              aria-label={sourcesCollapsed ? "Expand sources" : "Collapse sources"}
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
                  note={note}
                  transcript={transcript}
                  transcriptError={transcriptError}
                  onClose={() => setDetailSource(null)}
                />
              ) : (
                <SourcesListMode
                  listId={listId}
                  sources={sources}
                  onOpenSource={handleOpenSource}
                />
              )}
            </div>
          )}
        </div>

        <Resizer onMouseDown={onMouseDownLeft} />

        {/* ── Chat panel ── */}
        <div
          style={{ width: `${chatW}px`, minWidth: `${MIN_PANEL}px` }}
          className={panelBase}
        >
          <SkeletonPanel
            title="Chat"
            note="List-scoped chat arrives in v1. This panel stays intentionally quiet until then."
          />
        </div>

        <Resizer onMouseDown={onMouseDownRight} />

        {/* ── Lab panel ── */}
        <div
          style={{ width: `${labW}px`, minWidth: `${MIN_PANEL}px` }}
          className={panelBase}
        >
          <SkeletonPanel
            title="Lab"
            note="Synthesis tools and overview export arrive in v1."
          />
        </div>
      </div>
    </>
  );
}
