import { MdClose } from "react-icons/md";
import type { NoteContent, Source } from "@/lib/types";
import { NoteAccordion } from "@/components/lists/NoteAccordion";

export function SourcesViewerMode({
  source,
  note,
  transcript,
  transcriptError,
  onClose,
}: {
  source: Source;
  note: NoteContent | null;
  transcript: string | null;
  transcriptError: string | null;
  onClose: () => void;
}) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-start gap-3 border-b border-border px-4 py-4">
        <button
          type="button"
          onClick={onClose}
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
          aria-label="Close viewer"
        >
          <MdClose size={16} />
        </button>
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-sm font-medium text-ink">{source.title}</p>
          <p className="m-0 mt-0.5 text-xs text-muted">{source.platform}</p>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {note && <NoteAccordion markdown={note.markdown} />}

        <div className="space-y-2">
          {transcriptError && (
            <p className="text-xs text-rose-700">{transcriptError}</p>
          )}
          {transcript && (
            <>
              <p className="text-xs font-semibold uppercase tracking-wider text-muted/70">Transcript</p>
              <pre className="p-1 m-0 whitespace-pre-wrap font-mono text-xs text-muted leading-relaxed">
                {transcript}
              </pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
