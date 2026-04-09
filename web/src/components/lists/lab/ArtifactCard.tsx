import { AlertCircle, MoreVertical, X } from "lucide-react";

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
}

export function ArtifactCard({ artifact, onDismiss }: ArtifactCardProps) {
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
  return (
    <div className="group flex items-center gap-2 rounded-2xl border border-border bg-white/64 px-4 py-3 transition hover:bg-white hover:shadow-sm">
      <div className="min-w-0 flex-1">
        <p className="m-0 truncate text-sm font-bold text-ink">{artifact.name}</p>
        <p className="m-0 mt-0.5 text-xs text-muted">
          {artifact.type.replace(/_/g, " ")} · {artifact.source_ids.length} source{artifact.source_ids.length !== 1 ? "s" : ""} · {formatDate(artifact.created_at)}
        </p>
      </div>
      <ContextMenu
        items={[]}
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
