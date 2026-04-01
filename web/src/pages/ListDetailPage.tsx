import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { ChatPanel } from "../components/chat/ChatPanel";
import { StudioPanel } from "../components/studio/StudioPanel";
import { SourcesPanel } from "../components/sources/SourcesPanel";
import { api, toErrorMessage } from "../lib/api";
import { downloadTextFile } from "../lib/download";
import type { NoteContent, Source } from "../lib/types";

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

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const [lists, nextSources] = await Promise.all([api.listLists(), api.listSources(listId)]);
      if (cancelled) {
        return;
      }
      setListName(lists.find((entry) => entry.id === listId)?.name ?? "List workspace");
      setSources(nextSources);
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [listId]);

  async function handleIngest(url: string) {
    setIngestBusy(true);
    setIngestError(null);
    setIngestStatus(null);
    try {
      const result = await api.ingestUrl(listId, url);
      const queued = result.queued.length;
      const skipped = result.skipped.length;
      setIngestStatus(
        skipped > 0 ? `Queued ${queued} source and skipped ${skipped}.` : `Queued ${queued} source.`,
      );
      setSources(await api.listSources(listId));
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

  return (
    <div className="workspace">
      <section className="workspace-hero">
        <p className="home-hero__eyebrow">Notebook workspace</p>
        <h1 className="page-heading">{listName}</h1>
        <p className="page-lede">Queue sources, inspect notes, and export a list overview from one reading surface.</p>
      </section>
      <div className="workspace-grid">
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
