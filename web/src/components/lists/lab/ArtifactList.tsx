import { useCallback, useEffect, useMemo, useState } from "react";

import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import type { Artifact } from "@/lib/types";

interface ArtifactListProps {
  listId: string;
}

export function ArtifactList({ listId }: ArtifactListProps) {
  const { getJobs, dismissJob } = useJobActivity();
  const artifactJobs = useMemo(() => getJobs("artifact", listId), [getJobs, listId]);
  const [refreshedJobs, setRefreshedJobs] = useState<string[]>([]);

  const [artifacts, setArtifacts] = useState<Artifact[]>([]);

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

  return (
    <div className="space-y-2">
      {artifacts.map((artifact) => (
        <ArtifactCard
          key={artifact.id}
          artifact={artifact}
          onDismiss={artifact.status === "error" ? handleDismiss : undefined}
        />
      ))}
      {artifacts.length === 0 && (
        <p className="text-sm text-muted">No artifacts yet.</p>
      )}
    </div>
  );
}

// Import at bottom to avoid circular dependency
import { ArtifactCard } from "./ArtifactCard";
