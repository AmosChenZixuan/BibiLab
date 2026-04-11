import { useCallback, useEffect, useState } from "react";
import { ArrowRight, BookOpen, PenLine, Plus, Zap } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { Modal } from "@/components/ui/Modal";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import { ARTIFACT_TYPE_KEYS } from "@/lib/artifactTypes";
import { templates } from "@/lib/templates";
import type { ArtifactJob, ArtifactType } from "@/lib/types";

type ReportFormatType = "custom" | "brief" | "study_guide" | "blog_post";

interface ReportFormat {
  type: ReportFormatType;
  labelKey: string;
  descKey: string;
  icon: React.ReactNode;
}

const FORMAT_OPTIONS: ReportFormat[] = [
  { type: "custom", labelKey: "lab.reportsModal.custom", descKey: "lab.reportsModal.customDesc", icon: <Plus size={18} /> },
  { type: "brief", labelKey: ARTIFACT_TYPE_KEYS.brief, descKey: "lab.reportsModal.briefDesc", icon: <Zap size={18} /> },
  { type: "study_guide", labelKey: ARTIFACT_TYPE_KEYS.study_guide, descKey: "lab.reportsModal.studyGuideDesc", icon: <BookOpen size={18} /> },
  { type: "blog_post", labelKey: ARTIFACT_TYPE_KEYS.blog_post, descKey: "lab.reportsModal.blogPostDesc", icon: <PenLine size={18} /> },
];

interface ReportsModalProps {
  open: boolean;
  listId: string;
  sourceIds: string[];
  onClose: () => void;
  onArtifactGenerated: (artifactId: string, type: ArtifactType, sourceIds: string[]) => void;
}

export function ReportsModal({ open, listId, sourceIds, onClose, onArtifactGenerated }: ReportsModalProps) {
  const { t, lang } = useLanguage();
  const { trackJobs } = useJobActivity();

  const [selectedFormat, setSelectedFormat] = useState<ReportFormatType>("custom");
  const [prompt, setPrompt] = useState("");

  useEffect(() => {
    if (open) {
      setPrompt("");
      setSelectedFormat("custom");
    }
  }, [open]);

  const handleFormatSelect = useCallback(
    (format: ReportFormat) => {
      setSelectedFormat(format.type);
      const template = templates[format.type as keyof typeof templates];
      if (template) {
        setPrompt(template[lang === "zh" ? "zh" : "en"]);
      }
    },
    [lang],
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      const trimmedPrompt = prompt.trim();
      if (!trimmedPrompt) return;

      const artifactType: ArtifactType = selectedFormat === "custom" ? "custom_report" : selectedFormat;

      const job = await api.createArtifact(listId, {
        type: artifactType,
        prompt: trimmedPrompt,
        source_ids: sourceIds,
      }) as ArtifactJob;
      trackJobs([{ id: job.id, producer: "artifact", label: artifactType, contextKey: listId }]);
      onArtifactGenerated(job.meta.artifact_id ?? job.id, artifactType, sourceIds);
      onClose();
    },
    [listId, prompt, selectedFormat, sourceIds, onClose, trackJobs, onArtifactGenerated],
  );

  return (
    <Modal open={open} onClose={onClose} title={t("lab.reportsModal.title")} size="lg">
      <form onSubmit={handleSubmit} className="grid gap-5">
        <div className="grid gap-2.5">
          <span className="text-small font-semibold tracking-wide text-secondary-text uppercase">
            {t("lab.reportsModal.format")}
          </span>
          <div className="grid grid-cols-4 gap-2">
            {FORMAT_OPTIONS.map((format) => (
              <button
                key={format.type}
                type="button"
                onClick={() => handleFormatSelect(format)}
                className={`w-full flex flex-col items-center gap-1.5 rounded-2xl border p-3.5 text-center transition ${
                  selectedFormat === format.type
                    ? "border-pink bg-white shadow-sm"
                    : "border-divider/40 bg-white/64 hover:bg-white hover:shadow-sm"
                }`}
              >
                <span className="text-sky-blue">{format.icon}</span>
                <span className="text-caption font-medium text-charcoal">{t(format.labelKey)}</span>
                <span className="text-small text-secondary-text">{t(format.descKey)}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="grid gap-2">
          <span className="text-small font-semibold tracking-wide text-secondary-text uppercase">
            {t("lab.reportsModal.customPrompt")}
          </span>
          <div className="relative rounded-2xl border border-divider/40 bg-white/80 p-3 pr-10">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={t("lab.reportsModal.placeholder")}
              rows={12}
              className="w-full pr-3 resize-none bg-transparent text-caption text-charcoal placeholder:text-secondary-text/50 outline-none"
            />
            <button
              type="submit"
              disabled={!prompt.trim()}
              aria-label="Submit"
              className="absolute bottom-2.5 right-2.5 flex h-7 w-7 items-center justify-center rounded-full bg-pink text-white transition disabled:opacity-40 hover:bg-pink/80"
            >
              <ArrowRight size={15} />
            </button>
          </div>
        </div>
      </form>
    </Modal>
  );
}
