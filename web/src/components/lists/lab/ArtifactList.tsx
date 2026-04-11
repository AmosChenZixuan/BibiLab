import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import { downloadTextFile } from "@/lib/download";
import type { Artifact } from "@/lib/types";

import { Sparkles } from "lucide-react";

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
  const scrollRef = useRef<HTMLDivElement>(null);

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
      onArtifactsChange((prev) => prev.filter((a) => a.id !== artifactId));
    },
    [onArtifactsChange],
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

  const viewPromptArtifact = viewPromptArtifactId
    ? artifacts.find((a) => a.id === viewPromptArtifactId) ?? null
    : null;

  return (
    <div className="flex h-full flex-col">
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto"
        style={{ minHeight: 0 }}
      >
        <div className="space-y-2 pb-2">
          {artifacts.map((artifact) => (
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
                      const a = artifacts.find((art) => art.id === id);
                      if (a) onViewArtifact(a);
                    }
                  : undefined
              }
              onDelete={artifact.status === "completed" ? handleDelete : undefined}
            />
          ))}
          {artifacts.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
              <Sparkles size={32} className="text-charcoal" strokeWidth={1.5} />
              <span className="text-caption font-medium text-charcoal">{t("lab.artifactList.emptyTitle")}</span>
              <span className="text-small text-secondary-text">{t("lab.artifactList.emptyDesc")}</span>
            </div>
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
