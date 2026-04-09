import { useCallback, useEffect, useMemo, useState } from "react";

import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import { downloadTextFile } from "@/lib/download";
import type { Artifact } from "@/lib/types";

import { ArtifactCard } from "./ArtifactCard";
import { ViewPromptModal } from "./ViewPromptModal";

interface ArtifactListProps {
  listId: string;
}

export function ArtifactList({ listId }: ArtifactListProps) {
  const { getJobs, dismissJob } = useJobActivity();
  const artifactJobs = useMemo(() => getJobs("artifact", listId), [getJobs, listId]);
  const [refreshedJobs, setRefreshedJobs] = useState<string[]>([]);

  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [viewPromptArtifactId, setViewPromptArtifactId] = useState<string | null>(null);

  // Initial load
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const result = await api.listArtifacts(listId);
        if (cancelled) return;
        setArtifacts(result ?? []);
      } catch {
        // Non-critical: show empty list on error
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [listId]);

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
        setArtifacts(next ?? []);
        for (const { job } of completed) {
          setRefreshedJobs((prev) => [...prev, job.id]);
          await dismissJob(job.id);
        }
      } catch {
        // Non-critical: leave jobs in terminal state
      }
    }
    void refresh();
    return () => {
      cancelled = true;
    };
  }, [artifactJobs, listId, refreshedJobs, dismissJob]);

  const handleDismiss = useCallback(
    async (artifactId: string) => {
      setArtifacts((prev) => prev.filter((a) => a.id !== artifactId));
    },
    [],
  );

  const handleDownload = useCallback(async (artifactId: string) => {
    try {
      const result = await api.getArtifactContent(artifactId);
      if (!result) return;
      const artifact = artifacts.find((a) => a.id === artifactId);
      const filename = artifact ? `${artifact.name}.md` : "artifact.md";
      downloadTextFile(filename, result.content);
    } catch {
      // Silent failure - card stays
    }
  }, [artifacts]);

  const handleRename = useCallback((artifactId: string, name: string) => {
    let previousName: string;
    setArtifacts((prev) => {
      const artifact = prev.find((a) => a.id === artifactId);
      if (!artifact) return prev;
      previousName = artifact.name;
      return prev.map((a) => (a.id === artifactId ? { ...a, name } : a));
    });
    // API call (fire and forget, silent failure)
    void api.updateArtifact(artifactId, { name }).catch(() => {
      // Revert optimistic update on failure
      setArtifacts((prev) =>
        prev.map((a) => (a.id === artifactId ? { ...a, name: previousName } : a)),
      );
    });
  }, []);

  const handleDelete = useCallback(async (artifactId: string) => {
    try {
      await api.deleteArtifact(artifactId);
      setArtifacts((prev) => prev.filter((a) => a.id !== artifactId));
    } catch {
      // Silent failure - card stays
    }
  }, []);

  const viewPromptArtifact = viewPromptArtifactId
    ? artifacts.find((a) => a.id === viewPromptArtifactId) ?? null
    : null;

  return (
    <>
      <div className="space-y-2">
        {artifacts.map((artifact) => (
          <ArtifactCard
            key={artifact.id}
            artifact={artifact}
            onDismiss={artifact.status === "error" ? handleDismiss : undefined}
            onDownload={artifact.status === "done" ? handleDownload : undefined}
            onRename={artifact.status === "done" ? handleRename : undefined}
            onViewPrompt={artifact.status === "done" ? setViewPromptArtifactId : undefined}
            onDelete={artifact.status === "done" ? handleDelete : undefined}
          />
        ))}
        {artifacts.length === 0 && (
          <p className="text-sm text-muted">No artifacts yet.</p>
        )}
      </div>
      {viewPromptArtifact && (
        <ViewPromptModal
          open={true}
          onClose={() => setViewPromptArtifactId(null)}
          prompt={viewPromptArtifact.prompt}
        />
      )}
    </>
  );
}

// Import at bottom to avoid circular dependency
