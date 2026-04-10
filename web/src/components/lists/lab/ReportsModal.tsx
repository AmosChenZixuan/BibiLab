import { useCallback, useState } from "react";
import { ArrowRight, BookOpen, PenLine, Plus, Zap } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { Modal } from "@/components/ui/Modal";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { api } from "@/lib/api";
import { ARTIFACT_TYPE_KEYS } from "@/components/lists/lab/ArtifactCard";
import { templates } from "@/lib/templates";
import type { ArtifactJob, ArtifactType } from "@/lib/types";

interface ReportFormat {
  type: string;
  labelKey: string;
  descKey: string;
  icon: React.ReactNode;
  isCustom?: boolean;
}

const FORMAT_OPTIONS: ReportFormat[] = [
  { type: "custom", labelKey: "lab.reportsModal.custom", descKey: "lab.reportsModal.customDesc", icon: <Plus size={18} />, isCustom: true },
  { type: "brief", labelKey: ARTIFACT_TYPE_KEYS.brief, descKey: "lab.reportsModal.briefDesc", icon: <Zap size={18} /> },
  { type: "study_guide", labelKey: ARTIFACT_TYPE_KEYS.study_guide, descKey: "lab.reportsModal.studyGuideDesc", icon: <BookOpen size={18} /> },
  { type: "blog_post", labelKey: ARTIFACT_TYPE_KEYS.blog_post, descKey: "lab.reportsModal.blogPostDesc", icon: <PenLine size={18} /> },
];

const MAX_FORMAT_NAME_LENGTH = 50;

interface ReportsModalProps {
  open: boolean;
  listId: string;
  sourceIds: string[];
  onClose: () => void;
  onArtifactGenerated: (artifactId: string, type: ArtifactType) => void;
}

export function ReportsModal({ open, listId, sourceIds, onClose, onArtifactGenerated }: ReportsModalProps) {
  const { t, lang } = useLanguage();
  const { trackJobs } = useJobActivity();

  const [selectedFormat, setSelectedFormat] = useState<string>("custom");
  const [prompt, setPrompt] = useState("");
  const [customFormatName, setCustomFormatName] = useState(() => t("lab.reportsModal.custom"));
  const [isEditingCustomName, setIsEditingCustomName] = useState(false);

  const handleFormatSelect = useCallback(
    (format: ReportFormat) => {
      setSelectedFormat(format.type);

      if (format.isCustom) {
        // For custom format, use whatever is in the textarea (or empty)
        return;
      }

      // Fill textarea with template based on current language
      const templateKey = format.type as keyof typeof templates;
      if (templates[templateKey]) {
        setPrompt(templates[templateKey][lang === "zh" ? "zh" : "en"]);
      }
    },
    [lang],
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      const trimmedPrompt = prompt.trim();
      if (!trimmedPrompt) return;

      // Determine the artifact type
      const artifactType: ArtifactType = selectedFormat === "custom" ? customFormatName.trim() : (selectedFormat as ArtifactType);

      if (!artifactType) return;

      const job = await api.createArtifact(listId, {
        type: artifactType,
        prompt: trimmedPrompt,
        source_ids: sourceIds,
      }) as ArtifactJob;
      trackJobs([{ id: job.id, producer: "artifact", label: artifactType, contextKey: listId }]);
      onArtifactGenerated(job.meta.artifact_id ?? job.id, artifactType);

      // Reset state
      onClose();
      setPrompt("");
      setSelectedFormat("custom");
      setCustomFormatName(t("lab.reportsModal.custom"));
      setIsEditingCustomName(false);
    },
    [listId, prompt, selectedFormat, customFormatName, sourceIds, onClose, trackJobs, onArtifactGenerated, t],
  );

  const handleClose = useCallback(() => {
    onClose();
    setPrompt("");
    setSelectedFormat("custom");
    setCustomFormatName(t("lab.reportsModal.custom"));
    setIsEditingCustomName(false);
  }, [onClose, t]);

  return (
    <Modal open={open} onClose={handleClose} title={t("lab.reportsModal.title")} size="lg">
      <form onSubmit={handleSubmit} className="grid gap-5">
        {/* Format options */}
        <div className="grid gap-2.5">
          <span className="text-[11px] font-semibold tracking-wide text-muted uppercase">
            {t("lab.reportsModal.format")}
          </span>
          <div className="grid grid-cols-4 gap-2">
            {FORMAT_OPTIONS.map((format) => {
              const isCustomSelected = format.isCustom && selectedFormat === "custom";

              if (isCustomSelected) {
                return (
                  <div
                    key={format.type}
                    className="rounded-2xl border border-[#5b7faa] bg-white shadow-sm flex flex-col items-center gap-1.5 p-3.5 text-center"
                  >
                    <span className="text-[#5b7faa]">{format.icon}</span>
                    {isEditingCustomName ? (
                      <input
                        type="text"
                        value={customFormatName}
                        onChange={(e) => setCustomFormatName(e.target.value.slice(0, MAX_FORMAT_NAME_LENGTH))}
                        onBlur={() => setIsEditingCustomName(false)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            setIsEditingCustomName(false);
                          }
                        }}
                        autoFocus
                        className="max-w-[80px] bg-transparent text-[13px] text-ink outline-none text-center border border-[#5b7faa]"
                        placeholder={t("lab.reportsModal.custom")}
                      />
                    ) : (
                      <button
                        type="button"
                        onClick={() => setIsEditingCustomName(true)}
                        className="text-[13px] font-medium text-ink max-w-[80px] truncate hover:text-[#5b7faa]"
                      >
                        {customFormatName}
                      </button>
                    )}
                    <span className="text-[11px] text-muted">{t(format.descKey)}</span>
                  </div>
                );
              }

              return (
                <button
                  key={format.type}
                  type="button"
                  onClick={() => handleFormatSelect(format)}
                  className={`w-full flex flex-col items-center gap-1.5 rounded-2xl border p-3.5 text-center transition ${
                    selectedFormat === format.type
                      ? "border-[#5b7faa] bg-white shadow-sm"
                      : "border-border/40 bg-white/64 hover:bg-white hover:shadow-sm"
                  }`}
                >
                  <span className="text-[#5b7faa]">{format.icon}</span>
                  <span className="text-[13px] font-medium text-ink">
                    {format.isCustom ? customFormatName : t(format.labelKey)}
                  </span>
                  <span className="text-[11px] text-muted">{t(format.descKey)}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Prompt textarea */}
        <div className="grid gap-2">
          <span className="text-[11px] font-semibold tracking-wide text-muted uppercase">
            {t("lab.reportsModal.customPrompt")}
          </span>
          <div className="relative rounded-2xl border border-border/40 bg-white/80 p-3 pr-10">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={selectedFormat ? t("lab.reportsModal.placeholder") : t("lab.reportsModal.selectFormat")}
              rows={12}
              className="w-full pr-3 resize-none bg-transparent text-[13px] text-ink placeholder:text-muted/50 outline-none"
            />
            <button
              type="submit"
              disabled={!prompt.trim() || !selectedFormat}
              aria-label="Submit"
              className="absolute bottom-2.5 right-2.5 flex h-7 w-7 items-center justify-center rounded-full bg-[#5b7faa] text-white transition disabled:opacity-40 hover:bg-[#4a6d91]"
            >
              <ArrowRight size={15} />
            </button>
          </div>
        </div>
      </form>
    </Modal>
  );
}
