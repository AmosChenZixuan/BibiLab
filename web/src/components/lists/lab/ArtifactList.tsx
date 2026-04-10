import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import { downloadTextFile } from "@/lib/download";
import type { Artifact } from "@/lib/types";

import { ArtifactCard } from "./ArtifactCard";
import { ViewPromptModal } from "./ViewPromptModal";

interface ArtifactListProps {
  listId: string;
  artifacts: Artifact[];
  onArtifactsChange: (artifacts: Artifact[]) => void;
  onViewArtifact?: (artifact: Artifact) => void;
}

export function ArtifactList({ listId, artifacts, onArtifactsChange, onViewArtifact }: ArtifactListProps) {
  const { getJobs, dismissJob } = useJobActivity();
  const artifactJobs = useMemo(() => getJobs("artifact", listId), [getJobs, listId]);
  const [refreshedJobs, setRefreshedJobs] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  const [currentArtifacts, setCurrentArtifacts] = useState<Artifact[]>(artifacts);
  const [viewPromptArtifactId, setViewPromptArtifactId] = useState<string | null>(null);

  // Sync currentArtifacts when artifacts prop changes (e.g., after initial load)
  useEffect(() => {
    setCurrentArtifacts(artifacts);
  }, [artifacts]);

  // When a job flips to done, refresh artifacts and dismiss
  useEffect(() => {
    const completed = artifactJobs.filter(
      (item) => item.isTerminal && item.job.status === "done" && !refreshedJobs.includes(item.job.id),
    );
    if (completed.length === 0) return;

    let cancelled = false;
    async function refresh() {
      try {
        const next = await api.listArtifacts(listId);
        if (cancelled) return;
        setCurrentArtifacts(next ?? []);
        onArtifactsChange(next ?? []);
        await Promise.all(completed.map(({ job }) => dismissJob(job.id)));
        setRefreshedJobs((prev) => [...prev, ...completed.map(({ job }) => job.id)]);
      } catch {
        // Non-critical: leave jobs in terminal state
      }
    }
    void refresh();
    return () => {
      cancelled = true;
    };
  }, [artifactJobs, listId, refreshedJobs, dismissJob, onArtifactsChange]);

  const handleDismiss = useCallback(
    async (artifactId: string) => {
      setCurrentArtifacts((prev) => prev.filter((a) => a.id !== artifactId));
    },
    [],
  );

  const handleDownload = useCallback(async (artifactId: string) => {
    try {
      const result = await api.getArtifactContent(artifactId);
      if (!result) return;
      const artifact = currentArtifacts.find((a) => a.id === artifactId);
      const filename = artifact ? `${artifact.name}.md` : "artifact.md";
      downloadTextFile(filename, result.content);
    } catch {
      // Silent failure - card stays
    }
  }, [currentArtifacts]);

  const handleRename = useCallback((artifactId: string, name: string) => {
    let previousName: string;
    setCurrentArtifacts((prev) => {
      const artifact = prev.find((a) => a.id === artifactId);
      if (!artifact) return prev;
      previousName = artifact.name;
      return prev.map((a) => (a.id === artifactId ? { ...a, name } : a));
    });
    // API call (fire and forget, silent failure)
    void api.updateArtifact(artifactId, { name }).catch(() => {
      // Revert optimistic update on failure
      setCurrentArtifacts((prev) =>
        prev.map((a) => (a.id === artifactId ? { ...a, name: previousName } : a)),
      );
    });
  }, []);

  const handleDelete = useCallback(async (artifactId: string) => {
    try {
      await api.deleteArtifact(artifactId);
      setCurrentArtifacts((prev) => prev.filter((a) => a.id !== artifactId));
    } catch {
      // Silent failure - card stays
    }
  }, []);

  const viewPromptArtifact = viewPromptArtifactId
    ? currentArtifacts.find((a) => a.id === viewPromptArtifactId) ?? null
    : null;

  return (
    <div className="flex h-full flex-col">
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto"
        style={{ minHeight: 0 }}
      >
        <div className="space-y-2 pb-2">
          {currentArtifacts.map((artifact) => (
            <ArtifactCard
              key={artifact.id}
              artifact={artifact}
              onDismiss={artifact.status === "failed" ? handleDismiss : undefined}
              onDownload={artifact.status === "completed" ? handleDownload : undefined}
              onRename={artifact.status === "completed" ? handleRename : undefined}
              onViewPrompt={artifact.status === "completed" ? setViewPromptArtifactId : undefined}
              onView={
                artifact.status === "completed" && onViewArtifact
                  ? (id: string) => {
                      const a = currentArtifacts.find((art) => art.id === id);
                      if (a) onViewArtifact(a);
                    }
                  : undefined
              }
              onDelete={artifact.status === "completed" ? handleDelete : undefined}
            />
          ))}
          {currentArtifacts.length === 0 && (
            <p className="text-sm text-muted">No artifacts yet.</p>
          )}
        </div>
      </div>
      {viewPromptArtifact && (
        <ViewPromptModal
          open={true}
          onClose={() => setViewPromptArtifactId(null)}
          prompt={viewPromptArtifact.prompt}
        />
      )}
    </div>
  );
}
