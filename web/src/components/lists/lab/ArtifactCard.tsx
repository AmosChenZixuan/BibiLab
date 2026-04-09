import { AlertCircle, Download, FileText, MoreVertical, Pencil, Trash2, X } from "lucide-react";
import { useRef, useState } from "react";

import { ContextMenu } from "@/components/ui/ContextMenu";
import type { Artifact } from "@/lib/types";

function formatArtifactTypeLabel(type: Artifact["type"]): string {
  const labels: Record<Artifact["type"], string> = {
    brief: "BRIEF",
    study_guide: "STUDY_GUIDE",
    blog_post: "BLOG_POST",
    custom_report: "CUSTOM_REPORT",
  };
  return labels[type];
}

function formatDate(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

interface ArtifactCardProps {
  artifact: Artifact;
  onDismiss?: (artifactId: string) => void;
  onDownload?: (artifactId: string) => void;
  onRename?: (artifactId: string, name: string) => void;
  onViewPrompt?: (artifactId: string) => void;
  onDelete?: (artifactId: string) => void;
}

export function ArtifactCard({
  artifact,
  onDismiss,
  onDownload,
  onRename,
  onViewPrompt,
  onDelete,
}: ArtifactCardProps) {
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState(artifact.name);
  const inputRef = useRef<HTMLInputElement>(null);

  if (artifact.status === "generating") {
    return (
      <div className="flex flex-col gap-2.5 rounded-2xl border border-blue/20 bg-sky/6 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue/40" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-blue/70" />
          </span>
          <p className="m-0 min-w-0 flex-1 truncate text-sm font-medium text-ink">
            {formatArtifactTypeLabel(artifact.type)}
          </p>
        </div>
        <div data-testid="artifact-skeleton" className="space-y-1.5">
          <div className="h-3 w-3/4 rounded bg-blue/20" />
          <div className="h-3 w-1/2 rounded bg-blue/20" />
        </div>
      </div>
    );
  }

  if (artifact.status === "error") {
    return (
      <div className="flex items-start gap-3 rounded-2xl border border-pink/30 bg-pink/6 px-4 py-3">
        <AlertCircle data-testid="alert-icon" size={16} className="mt-0.5 shrink-0 text-pink" />
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-sm font-medium text-ink">{artifact.name}</p>
          {artifact.error && (
            <p className="m-0 mt-0.5 text-xs text-pink">{artifact.error}</p>
          )}
        </div>
        {onDismiss && (
          <button
            type="button"
            onClick={() => onDismiss(artifact.id)}
            aria-label="Dismiss"
            className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
          >
            <X size={14} />
          </button>
        )}
      </div>
    );
  }

  // done state
  const doneItems = [
    ...(onRename
      ? [
          {
            label: "Rename",
            icon: <Pencil size={14} />,
            onClick: () => {
              setRenameValue(artifact.name);
              setIsRenaming(true);
              // Focus input after render
              setTimeout(() => inputRef.current?.select(), 0);
            },
          },
        ]
      : []),
    ...(onDownload
      ? [
          {
            label: "Download",
            icon: <Download size={14} />,
            onClick: () => onDownload(artifact.id),
          },
        ]
      : []),
    ...(onViewPrompt
      ? [
          {
            label: "View Prompt",
            icon: <FileText size={14} />,
            onClick: () => onViewPrompt(artifact.id),
          },
        ]
      : []),
    ...(onDelete
      ? [
          {
            label: "Delete",
            icon: <Trash2 size={14} />,
            onClick: () => onDelete(artifact.id),
            variant: "danger" as const,
          },
        ]
      : []),
  ];

  function handleRenameSubmit() {
    const trimmed = renameValue.trim();
    if (trimmed && trimmed !== artifact.name) {
      onRename?.(artifact.id, trimmed);
    }
    setIsRenaming(false);
  }

  function handleRenameKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      handleRenameSubmit();
    } else if (e.key === "Escape") {
      setIsRenaming(false);
      setRenameValue(artifact.name);
    }
  }

  return (
    <div className="group flex items-center gap-2 rounded-2xl border border-border bg-white/64 px-4 py-3 transition hover:bg-white hover:shadow-sm">
      <div className="min-w-0 flex-1">
        {isRenaming ? (
          <input
            ref={inputRef}
            type="text"
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onBlur={handleRenameSubmit}
            onKeyDown={handleRenameKeyDown}
            className="m-0 w-full border border-border bg-white px-1 py-0 text-sm font-bold text-ink outline-none focus:border-blue"
            autoFocus
          />
        ) : (
          <p className="m-0 truncate text-sm font-bold text-ink">{artifact.name}</p>
        )}
        <p className="m-0 mt-0.5 text-xs text-muted">
          {artifact.type.replace(/_/g, " ")} · {artifact.source_ids.length} source
          {artifact.source_ids.length !== 1 ? "s" : ""} · {formatDate(artifact.created_at)}
        </p>
      </div>
      <ContextMenu
        items={doneItems}
        trigger={({ toggle, triggerRef }) => (
          <button
            ref={triggerRef}
            type="button"
            aria-label="Artifact options"
            onClick={toggle}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted opacity-0 transition group-hover:opacity-100 hover:bg-border hover:text-ink"
          >
            <MoreVertical size={16} />
          </button>
        )}
      />
    </div>
  );
}
