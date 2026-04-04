import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { MdExpandLess, MdExpandMore } from "react-icons/md";

export function NoteAccordion({ markdown }: { markdown: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-blue/25">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between gap-3 border-0 bg-transparent px-4 py-3.5 text-left transition hover:bg-sky/6"
      >
        <span className="text-sm font-semibold text-ink">Note</span>
        {expanded
          ? <MdExpandLess size={18} className="shrink-0 text-muted" />
          : <MdExpandMore size={18} className="shrink-0 text-muted" />}
      </button>
      {expanded && (
        <div className="border-t border-border px-4 py-4 space-y-2 text-sm text-muted [&_p]:text-sm [&_p]:text-muted [&_p]:leading-relaxed [&_p]:mb-2 [&_strong]:font-semibold [&_strong]:text-ink [&_h1]:text-sm [&_h1]:font-semibold [&_h1]:text-ink [&_h2]:text-xs [&_h2]:font-semibold [&_h2]:text-muted [&_h2]:uppercase [&_h2]:tracking-wider [&_ul]:pl-4 [&_ul]:space-y-1 [&_li]:text-sm [&_li]:text-muted [&_blockquote]:border-l-2 [&_blockquote]:border-blue/25 [&_blockquote]:pl-3 [&_blockquote]:italic [&_img]:max-w-full [&_img]:rounded-lg [&_img]:my-2">
          <ReactMarkdown>{markdown}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}
