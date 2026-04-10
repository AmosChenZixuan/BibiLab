import { useCallback, useState } from "react";
import { ArrowRight, BookOpen, PenLine, Zap } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { Modal } from "@/components/ui/Modal";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import type { ArtifactJob, ArtifactType } from "@/lib/types";

const SUGGESTED_PROMPTS: {
  type: ArtifactType;
  labelKey: string;
  descKey: string;
  icon: React.ReactNode;
}[] = [
  { type: "brief", labelKey: "lab.reportsModal.brief", descKey: "lab.reportsModal.briefDesc", icon: <Zap size={20} /> },
  { type: "study_guide", labelKey: "lab.reportsModal.studyGuide", descKey: "lab.reportsModal.studyGuideDesc", icon: <BookOpen size={20} /> },
  { type: "blog_post", labelKey: "lab.reportsModal.blogPost", descKey: "lab.reportsModal.blogPostDesc", icon: <PenLine size={20} /> },
];

interface ReportsModalProps {
  open: boolean;
  listId: string;
  sourceIds: string[];
  onClose: () => void;
  onArtifactGenerated: (artifactId: string, type: ArtifactType) => void;
}

export function ReportsModal({ open, listId, sourceIds, onClose, onArtifactGenerated }: ReportsModalProps) {
  const { t } = useLanguage();
  const [prompt, setPrompt] = useState("");
  const { trackJobs } = useJobActivity();

  const handleSubmit = useCallback(
    async (type: ArtifactType, promptText: string) => {
      const job = await api.createArtifact(listId, {
        type,
        prompt: promptText,
        source_ids: sourceIds,
      }) as ArtifactJob;
      trackJobs([{ id: job.id, producer: "artifact", label: type, contextKey: listId }]);
      onArtifactGenerated(job.meta.artifact_id ?? job.id, type);
      onClose();
      setPrompt("");
    },
    [listId, sourceIds, onClose, trackJobs, onArtifactGenerated],
  );

  const handleCustomSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = prompt.trim();
      if (!trimmed) return;
      void handleSubmit("custom_report", trimmed);
    },
    [prompt, handleSubmit],
  );

  return (
    <Modal open={open} onClose={onClose} title={t("lab.reportsModal.title")} size="md">
      <div className="grid gap-5">
        {/* Suggested prompts */}
        <div className="grid gap-2.5">
          <span className="text-[11px] font-semibold tracking-wide text-muted uppercase">
            {t("lab.reportsModal.suggested")}
          </span>
          <div className="grid grid-cols-3 gap-2.5">
            {SUGGESTED_PROMPTS.map((s) => (
              <button
                key={s.type}
                type="button"
                onClick={() => void handleSubmit(s.type, t(s.labelKey))}
                className="flex flex-col items-center gap-1.5 rounded-2xl border border-border/40 bg-white/64 p-3.5 text-center transition hover:bg-white hover:shadow-sm"
              >
                <span className="text-[#5b7faa]">{s.icon}</span>
                <span className="text-[13px] font-medium text-ink">{t(s.labelKey)}</span>
                <span className="text-[11px] text-muted">{t(s.descKey)}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Custom prompt */}
        <div className="grid gap-2">
          <span className="text-[11px] font-semibold tracking-wide text-muted uppercase">
            {t("lab.reportsModal.customPrompt")}
          </span>
          <form onSubmit={handleCustomSubmit}>
            <div className="relative rounded-2xl border border-border/40 bg-white/80 p-3 pr-10">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder={t("lab.reportsModal.placeholder")}
                rows={4}
                className="w-full resize-none bg-transparent text-[13px] text-ink placeholder:text-muted/50 outline-none"
              />
              <button
                type="submit"
                disabled={!prompt.trim()}
                aria-label="Submit"
                className="absolute bottom-2.5 right-2.5 flex h-7 w-7 items-center justify-center rounded-full bg-[#5b7faa] text-white transition disabled:opacity-40 hover:bg-[#4a6d91]"
              >
                <ArrowRight size={15} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </Modal>
  );
}
