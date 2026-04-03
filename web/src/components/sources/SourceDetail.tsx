import { useState } from "react";

import ReactMarkdown from "react-markdown";

import { downloadTextFile } from "../../lib/download";
import type { NoteContent, Source } from "../../lib/types";
import { Button, PanelBody } from "../../components/ui";

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
    <PanelBody>
      <div className="grid gap-3">
        <Button variant="ghost" onClick={onBack} type="button">
          Back to sources
        </Button>
        <div>
          <h3 className="m-0 font-serif text-2xl">{source.title}</h3>
          <p className="m-0 text-muted">{source.platform}</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2.5">
        <button
          className={`rounded-full border px-3.5 py-2.5 transition ${activeTab === "note" ? "border-transparent bg-blue text-white" : "border-border bg-sky/10 text-muted/80"}`}
          onClick={() => void onSelectTab("note")}
          type="button"
        >
          Note
        </button>
        <button
          className={`rounded-full border px-3.5 py-2.5 transition ${activeTab === "transcript" ? "border-transparent bg-blue text-white" : "border-border bg-sky/10 text-muted/80"}`}
          onClick={() => void onSelectTab("transcript")}
          type="button"
        >
          Transcript
        </button>
        <Button variant="ghost" disabled={downloading || !note} onClick={handleDownload} type="button">
          Download note
        </Button>
      </div>

      {activeTab === "note" ? (
        <div className="min-h-80 rounded-2xl border border-border bg-white/60 p-4.5">
          <ReactMarkdown>{note?.markdown ?? ""}</ReactMarkdown>
        </div>
      ) : (
        <div className="min-h-80 rounded-2xl border border-border bg-white/60 p-4.5">
          {transcriptLoading ? <p className="m-0 text-muted">Loading transcript...</p> : null}
          {transcriptError ? <p className="m-0 text-sm text-rose-900">{transcriptError}</p> : null}
          {transcript ? <pre className="m-0 whitespace-pre-wrap font-mono text-slate-600">{transcript}</pre> : null}
        </div>
      )}
    </PanelBody>
  );
}
