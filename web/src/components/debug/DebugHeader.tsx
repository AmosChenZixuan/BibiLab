import { Clipboard, X } from "lucide-react";

export function DebugHeader({
  messageId,
  model,
  timestamp,
  view,
  setView,
  onClose,
  dump,
}: {
  messageId: string;
  model?: string;
  timestamp?: string;
  view: "styled" | "raw";
  setView: (v: "styled" | "raw") => void;
  onClose: () => void;
  dump: unknown;
}) {
  return (
    <div className="h-16 flex items-center justify-between px-6 border-b border-(--color-border) bg-white">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-2xs font-medium px-2 py-0.5 rounded-full bg-pink-50 text-pink-900">
          DEBUG
        </span>
        <span className="text-sm text-(--color-muted) font-mono truncate">
          {messageId}
        </span>
        {timestamp && (
          <>
            <span className="text-(--color-muted)">·</span>
            <span className="text-sm text-(--color-muted)">{timestamp}</span>
          </>
        )}
        {model && (
          <>
            <span className="text-(--color-muted)">·</span>
            <span className="text-sm text-(--color-muted)">{model}</span>
          </>
        )}
      </div>
      <div className="flex items-center gap-1">
        <div className="inline-flex bg-(--color-surface) rounded-full p-0.5 text-xs font-medium">
          <button
            className={`px-3 py-1 rounded-full ${view === "styled" ? "bg-white shadow-sm" : "text-(--color-muted)"}`}
            onClick={() => setView("styled")}
          >
            Styled
          </button>
          <button
            className={`px-3 py-1 rounded-full ${view === "raw" ? "bg-white shadow-sm" : "text-(--color-muted)"}`}
            onClick={() => setView("raw")}
          >
            Raw
          </button>
        </div>
        <button
          className="w-10 h-10 rounded-full hover:bg-(--color-surface) flex items-center justify-center text-(--color-muted)"
          onClick={() =>
            navigator.clipboard?.writeText(JSON.stringify(dump, null, 2))
          }
          title="Copy dump JSON"
        >
          <Clipboard size={20} />
        </button>
        <button
          className="w-10 h-10 rounded-full hover:bg-(--color-surface) flex items-center justify-center text-(--color-muted)"
          onClick={onClose}
          title="Close (Esc)"
        >
          <X size={20} />
        </button>
      </div>
    </div>
  );
}
