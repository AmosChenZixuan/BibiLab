import { useCallback, useState } from "react";
import { ArrowRight } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { Modal } from "@/components/ui/Modal";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import type { ArtifactType } from "@/lib/types";

const SUGGESTED_PROMPTS: { type: ArtifactType; labelKey: string }[] = [
  { type: "brief", labelKey: "lab.reportsModal.brief" },
  { type: "study_guide", labelKey: "lab.reportsModal.studyGuide" },
  { type: "blog_post", labelKey: "lab.reportsModal.blogPost" },
];

interface ReportsModalProps {
  open: boolean;
  listId: string;
  sourceIds: string[];
  onClose: () => void;
}

export function ReportsModal({ open, listId, sourceIds, onClose }: ReportsModalProps) {
  const { t } = useLanguage();
  const [prompt, setPrompt] = useState("");
  const { trackJobs } = useJobActivity();

  const handleSubmit = useCallback(
    async (type: ArtifactType, promptText: string) => {
      const job = await api.createArtifact(listId, {
        type,
        prompt: promptText,
        source_ids: sourceIds,
      });
      trackJobs([{ id: job.id, producer: "artifact", label: type, contextKey: listId }]);
      onClose();
      setPrompt("");
    },
    [listId, sourceIds, onClose, trackJobs],
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
        <div className="grid grid-cols-3 gap-3">
          {SUGGESTED_PROMPTS.map((s) => (
            <button
              key={s.type}
              type="button"
              onClick={() => void handleSubmit(s.type, t(s.labelKey))}
              className="rounded-xl border border-border bg-white/64 px-3 py-3 text-center text-sm font-medium text-ink transition hover:bg-white hover:shadow-sm"
            >
              {t(s.labelKey)}
            </button>
          ))}
        </div>

        <form className="relative" onSubmit={handleCustomSubmit}>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={t("lab.reportsModal.placeholder")}
            className="w-full rounded-full border border-border bg-white/80 py-2.5 pr-10 pl-4 text-sm text-ink placeholder:text-muted/50 outline-none focus:border-blue/40 focus:bg-white transition"
          />
          <button
            type="submit"
            disabled={!prompt.trim()}
            aria-label="Submit"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 flex h-7 w-7 items-center justify-center rounded-full text-muted transition disabled:opacity-0 enabled:hover:bg-blue enabled:hover:text-white enabled:hover:shadow-sm"
          >
            <ArrowRight size={15} />
          </button>
        </form>
      </div>
    </Modal>
  );
}
