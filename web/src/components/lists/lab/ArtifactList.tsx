import { useCallback, useEffect, useMemo, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import { downloadTextFile } from "@/lib/download";
import type { Artifact, ArtifactJob } from "@/lib/types";

import { Sparkles } from "lucide-react";

import { ARTIFACT_TYPE_KEYS } from "@/lib/artifactTypes";
import { ArtifactCard } from "./ArtifactCard";
import { ViewPromptModal } from "./ViewPromptModal";

type ArtifactsUpdater = (prev: Artifact[]) => Artifact[];

interface ArtifactListProps {
  listId: string;
  artifacts: Artifact[];
  onArtifactsChange: (updater: ArtifactsUpdater) => void;
  onViewArtifact?: (artifact: Artifact) => void;
}

export function ArtifactList({ listId, artifacts, onArtifactsChange, onViewArtifact }: ArtifactListProps) {
  const { t } = useLanguage();
  const { getJobs, dismissJob } = useJobActivity();
  const artifactJobs = useMemo(() => getJobs("artifact", listId), [getJobs, listId]);
  const [refreshedJobs, setRefreshedJobs] = useState<string[]>([]);
  const [viewPromptArtifactId, setViewPromptArtifactId] = useState<string | null>(null);

  const artifactIds = useMemo(() => new Set(artifacts.map((a) => a.id)), [artifacts]);

  const generatingArtifacts = useMemo((): Artifact[] => {
    return artifactJobs
      .filter((item) => {
        const meta = item.job.meta as ArtifactJob["meta"];
        return !artifactIds.has(meta.artifact_id ?? "");
      })
      .filter((item) => !item.isTerminal)
      .map((item) => {
        const meta = item.job.meta as ArtifactJob["meta"];
        return {
          id: meta.artifact_id ?? item.job.id,
          name: t(ARTIFACT_TYPE_KEYS[item.label] ?? "lab.reportsModal.custom"),
          type: item.label as Artifact["type"],
          prompt: "",
          source_ids: [],
          status: "generating" as const,
          created_at: item.job.created_at,
        };
      });
  }, [artifactJobs, artifactIds, t]);

  // When a job flips to done, refresh artifacts and dismiss
  useEffect(() => {
    const refreshedSet = new Set(refreshedJobs);
    const completed = artifactJobs.filter(
      (item) => item.isTerminal && item.job.status === "done" && !refreshedSet.has(item.job.id),
    );
    if (completed.length === 0) return;

    let cancelled = false;
    async function refresh() {
      try {
        const next = await api.listArtifacts(listId);
        if (cancelled) return;
        onArtifactsChange(() => next ?? []);
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
      const job = artifactJobs.find((item) => {
        const meta = item.job.meta as ArtifactJob["meta"];
        return meta.artifact_id === artifactId;
      });
      if (job) {
        await dismissJob(job.job.id);
      }
      try {
        await api.deleteArtifact(artifactId);
      } catch {
        // Non-critical: artifact may not exist in DB yet
      }
      onArtifactsChange((prev) => prev.filter((a) => a.id !== artifactId));
    },
    [artifactJobs, dismissJob, onArtifactsChange],
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
    let previousName: string | undefined;
    onArtifactsChange((prev) => {
      const artifact = prev.find((a) => a.id === artifactId);
      if (!artifact) return prev;
      previousName = artifact.name;
      return prev.map((a) => (a.id === artifactId ? { ...a, name } : a));
    });
    void api.updateArtifact(artifactId, { name }).catch(() => {
      onArtifactsChange((prev) =>
        prev.map((a) => (a.id === artifactId ? { ...a, name: previousName ?? a.name } : a)),
      );
    });
  }, [onArtifactsChange]);

  const handleDelete = useCallback(async (artifactId: string) => {
    try {
      await api.deleteArtifact(artifactId);
      onArtifactsChange((prev) => prev.filter((a) => a.id !== artifactId));
    } catch {
      // Silent failure - card stays
    }
  }, [onArtifactsChange]);

  const allArtifacts = useMemo(
    () => [...generatingArtifacts, ...artifacts],
    [generatingArtifacts, artifacts],
  );

  const viewPromptArtifact = viewPromptArtifactId
    ? artifacts.find((a) => a.id === viewPromptArtifactId) ?? null
    : null;

  return (
    <div className="flex h-full flex-col space-y-2">
      {allArtifacts.map((artifact) => (
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
                  const a = allArtifacts.find((art) => art.id === id);
                  if (a) onViewArtifact(a);
                }
              : undefined
          }
          onDelete={artifact.status === "completed" ? handleDelete : undefined}
        />
      ))}
      {allArtifacts.length === 0 && (
        <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
          <Sparkles size={32} className="text-ink" strokeWidth={1.5} />
          <span className="text-[13px] font-medium text-ink">{t("lab.artifactList.emptyTitle")}</span>
          <span className="text-[11px] text-muted">{t("lab.artifactList.emptyDesc")}</span>
        </div>
      )}
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
