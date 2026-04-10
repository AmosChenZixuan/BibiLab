import { AlertCircle, Download, Eye, FileText, MoreVertical, Pencil, Trash2, X } from "lucide-react";
import { useRef, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { ContextMenu } from "@/components/ui/ContextMenu";
import type { Artifact } from "@/lib/types";

function formatArtifactTypeLabel(type: Artifact["type"], t: (key: string) => string): string {
  const labels: Record<string, string> = {
    brief: t("lab.artifactType.brief"),
    study_guide: t("lab.artifactType.studyGuide"),
    blog_post: t("lab.artifactType.blogPost"),
    custom_report: t("lab.artifactType.customReport"),
  };
  return labels[type] ?? type; // Fallback to type itself for custom types
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
  onView?: (artifactId: string) => void;
  onDelete?: (artifactId: string) => void;
}

export function ArtifactCard({
  artifact,
  onDismiss,
  onDownload,
  onRename,
  onViewPrompt,
  onView,
  onDelete,
}: ArtifactCardProps) {
  const { t } = useLanguage();
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
            {formatArtifactTypeLabel(artifact.type, t)}
          </p>
        </div>
        <div data-testid="artifact-skeleton" className="space-y-1.5">
          <div className="h-3 w-3/4 rounded bg-blue/20" />
          <div className="h-3 w-1/2 rounded bg-blue/20" />
        </div>
      </div>
    );
  }

  if (artifact.status === "failed") {
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
  const doneItems: {
    label: string;
    icon: React.ReactNode;
    onClick: () => void;
    variant?: "danger";
  }[] = [];
  if (onView) {
    doneItems.push({ label: t("lab.artifactCard.open"), icon: <Eye size={14} />, onClick: () => onView(artifact.id) });
  }
  if (onRename) {
    doneItems.push({
      label: t("lab.artifactCard.rename"),
      icon: <Pencil size={14} />,
      onClick: () => {
        setRenameValue(artifact.name);
        setIsRenaming(true);
        setTimeout(() => inputRef.current?.select(), 0);
      },
    });
  }
  if (onDownload) {
    doneItems.push({ label: t("lab.artifactCard.download"), icon: <Download size={14} />, onClick: () => onDownload(artifact.id) });
  }
  if (onViewPrompt) {
    doneItems.push({ label: t("lab.artifactCard.viewPrompt"), icon: <FileText size={14} />, onClick: () => onViewPrompt(artifact.id) });
  }
  if (onDelete) {
    doneItems.push({ label: t("lab.artifactCard.delete"), icon: <Trash2 size={14} />, onClick: () => onDelete(artifact.id), variant: "danger" });
  }

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

  function handleCardClick() {
    if (!isRenaming) {
      onView?.(artifact.id);
    }
  }

  return (
    <div className="group flex items-center gap-2 rounded-2xl border border-border bg-white/64 px-4 py-3 transition hover:bg-white hover:shadow-sm">
      {/* Clickable card content */}
      <div
        className="min-w-0 flex-1 cursor-pointer"
        onClick={handleCardClick}
      >
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
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <p className="m-0 truncate text-sm font-bold text-ink">{artifact.name}</p>
        )}
        <p className="m-0 mt-0.5 text-xs text-muted">
          {formatArtifactTypeLabel(artifact.type, t)} · {artifact.source_ids.length} source
          {artifact.source_ids.length !== 1 ? "s" : ""} · {formatDate(artifact.created_at)}
        </p>
      </div>

      {/* Options button */}
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
