import { useState } from "react";

import ReactMarkdown from "react-markdown";

import { downloadTextFile } from "../../lib/download";
import type { NoteContent, Source } from "../../lib/types";
import { ghostButtonClass, mutedTextClass, statusErrorClass, workspacePanelBodyClass } from "../../lib/ui";

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
    <div className={workspacePanelBodyClass}>
      <div className="grid gap-3">
        <button className={ghostButtonClass} onClick={onBack} type="button">
          Back to sources
        </button>
        <div>
          <h3 className="m-0 font-serif text-2xl">{source.title}</h3>
          <p className={mutedTextClass}>{source.platform}</p>
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
        <button className={ghostButtonClass} disabled={downloading || !note} onClick={handleDownload} type="button">
          Download note
        </button>
      </div>

      {activeTab === "note" ? (
        <div className="min-h-[320px] rounded-2xl border border-border bg-white/60 p-4.5">
          <ReactMarkdown>{note?.markdown ?? ""}</ReactMarkdown>
        </div>
      ) : (
        <div className="min-h-[320px] rounded-2xl border border-border bg-white/60 p-4.5">
          {transcriptLoading ? <p className={mutedTextClass}>Loading transcript...</p> : null}
          {transcriptError ? <p className={statusErrorClass}>{transcriptError}</p> : null}
          {transcript ? <pre className="m-0 whitespace-pre-wrap font-mono text-[#4e6485]">{transcript}</pre> : null}
        </div>
      )}
    </div>
  );
}
