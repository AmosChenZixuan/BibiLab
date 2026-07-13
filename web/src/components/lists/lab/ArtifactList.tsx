import { useCallback, useMemo, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { useDismissOnDone } from "@/components/jobs/useDismissOnDone";
import { api } from "@/lib/api";
import { downloadTextFile } from "@/lib/download";
import { usePendingDeletions } from "@/lib/hooks/usePendingDeletions";
import type { Artifact, ArtifactJob, ArtifactStatus } from "@/lib/types";

import { Sparkles } from "lucide-react";

import { ARTIFACT_TYPE_KEYS } from "@/lib/artifact-types";
import { ArtifactCard } from "./ArtifactCard";
import { ViewPromptModal } from "./ViewPromptModal";

function getArtifactJobMeta(item: { job: { meta: unknown } }): ArtifactJob["meta"] {
  return item.job.meta as ArtifactJob["meta"];
}

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
  const [viewPromptArtifactId, setViewPromptArtifactId] = useState<string | null>(null);
  const { isPending, run } = usePendingDeletions();

  const artifactIds = useMemo(() => new Set(artifacts.map((a) => a.id)), [artifacts]);

  const jobDerivedArtifacts = useMemo((): Artifact[] => {
    const result: Artifact[] = [];
    for (const item of artifactJobs) {
      const meta = getArtifactJobMeta(item);
      if (artifactIds.has(meta.artifact_id ?? "")) continue;
      if (item.isTerminal && item.job.status !== "failed") continue;
      const status: ArtifactStatus = item.job.status === "failed" ? "failed" : "generating";
      result.push({
        id: meta.artifact_id ?? item.job.id,
        name: t(ARTIFACT_TYPE_KEYS[item.label] ?? "lab.reportsModal.custom"),
        type: item.label as Artifact["type"],
        prompt: "",
        source_ids: [],
        status,
        error: item.job.error ?? undefined,
        created_at: item.job.created_at,
      });
    }
    return result;
  }, [artifactJobs, artifactIds, t]);

  // When an artifact job flips to done, refetch the artifact list and dismiss.
  useDismissOnDone({
    jobs: artifactJobs,
    onDone: async () => {
      const next = await api.listArtifacts(listId);
      onArtifactsChange(() => next ?? []);
    },
  });

  const handleDismiss = useCallback(
    async (artifactId: string) => {
      const job = artifactJobs.find((item) => getArtifactJobMeta(item).artifact_id === artifactId);
      if (job) {
        await dismissJob(job.job.id);
      }
    },
    [artifactJobs, dismissJob],
  );

  const handleDownload = useCallback(async (artifactId: string) => {
    try {
      const result = await api.getArtifactContent(artifactId);
      if (!result) return;
      const artifact = artifacts.find((a) => a.id === artifactId);
      const filename = artifact ? `${artifact.name || artifact.type}.md` : "artifact.md";
      downloadTextFile(filename, result.content);
    } catch {
      // Silent failure - card stays
    }
  }, [artifacts]);

  const handleRename = useCallback((artifactId: string, name: string) => {
    let previousName: string | null | undefined;
    onArtifactsChange((prev) => {
      const artifact = prev.find((a) => a.id === artifactId);
      if (!artifact) return prev;
      previousName = artifact.name;
      return prev.map((a) => (a.id === artifactId ? { ...a, name } : a));
    });
    void api.updateArtifact(artifactId, { name }).catch(() => {
      onArtifactsChange((prev) =>
        prev.map((a) => (a.id === artifactId ? { ...a, name: previousName !== undefined ? previousName : a.name } : a)),
      );
    });
  }, [onArtifactsChange]);

  const handleDelete = useCallback(async (artifactId: string) => {
    await run(artifactId, () => api.deleteArtifact(artifactId));
    onArtifactsChange((prev) => prev.filter((a) => a.id !== artifactId));
  }, [run, onArtifactsChange]);

  const allArtifacts = useMemo(
    () => jobDerivedArtifacts.concat(artifacts),
    [jobDerivedArtifacts, artifacts],
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
          isDeleting={isPending(artifact.id)}
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
