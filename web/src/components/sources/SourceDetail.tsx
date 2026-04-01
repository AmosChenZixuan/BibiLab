import { useState } from "react";

import ReactMarkdown from "react-markdown";

import { downloadTextFile } from "../../lib/download";
import type { NoteContent, Source } from "../../lib/types";

type Props = {
  activeTab: "note" | "transcript";
  note: NoteContent | null;
  source: Source;
  transcript: string | null;
  transcriptError: string | null;
  transcriptLoading: boolean;
  onBack: () => void;
  onSelectTab: (tab: "note" | "transcript") => Promise<void>;
};

export function SourceDetail({
  activeTab,
  note,
  onBack,
  onSelectTab,
  source,
  transcript,
  transcriptError,
  transcriptLoading,
}: Props) {
  const [downloading, setDownloading] = useState(false);

  function handleDownload() {
    if (!note) {
      return;
    }
    setDownloading(true);
    downloadTextFile(`${source.video_id}.md`, note.markdown);
    setDownloading(false);
  }

  return (
    <div className="workspace-panel__body">
      <div className="source-detail__header">
        <button className="ghost-button" onClick={onBack} type="button">
          Back to sources
        </button>
        <div>
          <h3 className="source-detail__title">{source.title}</h3>
          <p className="page-lede">{source.platform}</p>
        </div>
      </div>

      <div className="source-detail__tabs">
        <button
          className={`nav-link${activeTab === "note" ? " active" : ""}`}
          onClick={() => void onSelectTab("note")}
          type="button"
        >
          Note
        </button>
        <button
          className={`nav-link${activeTab === "transcript" ? " active" : ""}`}
          onClick={() => void onSelectTab("transcript")}
          type="button"
        >
          Transcript
        </button>
        <button className="ghost-button" disabled={downloading || !note} onClick={handleDownload} type="button">
          Download note
        </button>
      </div>

      {activeTab === "note" ? (
        <div className="markdown-panel">
          <ReactMarkdown>{note?.markdown ?? ""}</ReactMarkdown>
        </div>
      ) : (
        <div className="transcript-panel">
          {transcriptLoading ? <p className="page-lede">Loading transcript...</p> : null}
          {transcriptError ? <p className="status-message error">{transcriptError}</p> : null}
          {transcript ? <pre>{transcript}</pre> : null}
        </div>
      )}
    </div>
  );
}
