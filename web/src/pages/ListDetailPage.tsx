import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { ChatPanel } from "../components/chat/ChatPanel";
import { StudioPanel } from "../components/studio/StudioPanel";
import { SourcesPanel } from "../components/sources/SourcesPanel";
import { api, notifyJobsChanged, toErrorMessage } from "../lib/api";
import { downloadTextFile } from "../lib/download";
import type { NoteContent, Source } from "../lib/types";

function formatCount(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function ListDetailPage() {
  const { listId = "" } = useParams();
  const [listName, setListName] = useState("List workspace");
  const [sources, setSources] = useState<Source[]>([]);
  const [detailSource, setDetailSource] = useState<Source | null>(null);
  const [note, setNote] = useState<NoteContent | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"note" | "transcript">("note");
  const [ingestBusy, setIngestBusy] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [studioBusy, setStudioBusy] = useState(false);
  const [studioError, setStudioError] = useState<string | null>(null);
  const [studioStatus, setStudioStatus] = useState<string | null>(null);
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [lists, nextSources] = await Promise.all([api.listLists(), api.listSources(listId)]);
        if (cancelled) {
          return;
        }
        const currentName = lists.find((entry) => entry.id === listId)?.name ?? "List workspace";
        setListName(currentName);
        setDraftName(currentName);
        setSources(nextSources);
        setLoadError(null);
      } catch (error) {
        if (!cancelled) {
          setLoadError(toErrorMessage(error));
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [listId]);

  async function handleIngest(url: string, rerun: boolean) {
    setIngestBusy(true);
    setIngestError(null);
    setIngestStatus(null);
    try {
      const result = await api.ingestUrl(listId, url, rerun);
      const queuedSummary = `Queued ${formatCount(result.queued.length, "source")}`;
      const skippedSummary =
        result.skipped.length > 0 ? ` and skipped ${formatCount(result.skipped.length, "source")}` : "";
      setIngestStatus(`${queuedSummary}${skippedSummary}.`);
      setSources(await api.listSources(listId));
      notifyJobsChanged();
    } catch (error) {
      setIngestError(toErrorMessage(error));
    } finally {
      setIngestBusy(false);
    }
  }

  async function handleOpen(source: Source) {
    setDetailSource(source);
    setActiveTab("note");
    setTranscript(null);
    setTranscriptError(null);
    setNote(await api.getNoteContent(source.video_id));
  }

  async function handleDelete(source: Source) {
    if (!window.confirm(`Delete source "${source.title}"?`)) {
      return;
    }
    await api.deleteSource(listId, source.video_id);
    const nextSources = sources.filter((entry) => entry.video_id !== source.video_id);
    setSources(nextSources);
    if (detailSource?.video_id === source.video_id) {
      setDetailSource(null);
      setNote(null);
      setTranscript(null);
      setTranscriptError(null);
    }
  }

  async function handleSelectTab(tab: "note" | "transcript") {
    setActiveTab(tab);
    if (tab === "transcript" && detailSource && !transcript && !transcriptLoading) {
      setTranscriptLoading(true);
      setTranscriptError(null);
      try {
        const response = await api.getNoteTranscript(detailSource.video_id);
        setTranscript(response.text);
      } catch (error) {
        setTranscriptError(toErrorMessage(error));
      } finally {
        setTranscriptLoading(false);
      }
    }
  }

  async function handleGenerateOverview() {
    setStudioBusy(true);
    setStudioError(null);
    setStudioStatus(null);
    try {
      const overview = await api.generateOverview(listId);
      downloadTextFile(overview.filename, overview.content);
      setStudioStatus("Overview downloaded.");
    } catch (error) {
      setStudioError(toErrorMessage(error));
    } finally {
      setStudioBusy(false);
    }
  }

  async function handleRenameCommit() {
    const trimmed = draftName.trim();
    setEditingName(false);
    if (!trimmed || trimmed === listName) {
      setDraftName(listName);
      return;
    }
    setRenameError(null);
    try {
      const updated = await api.updateList(listId, trimmed);
      setListName(updated.name);
      setDraftName(updated.name);
    } catch (error) {
      setDraftName(listName);
      setRenameError(toErrorMessage(error));
    }
  }

  return (
    <div className="workspace">
      <section className="workspace-hero">
        <p className="home-hero__eyebrow">Notebook workspace</p>
        {editingName ? (
          <label className="field workspace-hero__rename">
            <span className="sr-only">List name</span>
            <input
              aria-label="List name"
              autoFocus
              className="workspace-hero__name-input"
              onBlur={() => void handleRenameCommit()}
              onChange={(event) => setDraftName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.currentTarget.blur();
                }
                if (event.key === "Escape") {
                  setDraftName(listName);
                  setEditingName(false);
                }
              }}
              value={draftName}
            />
          </label>
        ) : (
          <div className="workspace-hero__title-row">
            <h1 className="page-heading">{listName}</h1>
            <button
              aria-label="Edit list name"
              className="ghost-button"
              onClick={() => {
                setDraftName(listName);
                setEditingName(true);
                setRenameError(null);
              }}
              type="button"
            >
              Rename
            </button>
          </div>
        )}
        <p className="page-lede">Queue sources, inspect notes, and export a list overview from one reading surface.</p>
        {renameError ? <p className="status-message error">{renameError}</p> : null}
      </section>
      <div className="workspace-grid">
        {loadError ? (
          <section className="workspace-panel">
            <h2 className="workspace-panel__title">Sources</h2>
            <div className="workspace-panel__body">
              <p className="status-message error">{loadError}</p>
            </div>
          </section>
        ) : (
          <SourcesPanel
            activeTab={activeTab}
            detailSource={detailSource}
            ingestBusy={ingestBusy}
            ingestError={ingestError}
            ingestStatus={ingestStatus}
            note={note}
            onBack={() => {
              setDetailSource(null);
              setNote(null);
              setTranscript(null);
              setTranscriptError(null);
            }}
            onDelete={handleDelete}
            onIngest={handleIngest}
            onOpen={handleOpen}
            onSelectTab={handleSelectTab}
            sources={sources}
            transcript={transcript}
            transcriptError={transcriptError}
            transcriptLoading={transcriptLoading}
          />
        )}
        <ChatPanel />
        <StudioPanel
          busy={studioBusy}
          error={studioError}
          onGenerate={handleGenerateOverview}
          status={studioStatus}
        />
      </div>
    </div>
  );
}
