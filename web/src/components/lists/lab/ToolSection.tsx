import { useState } from "react";
import { Brain, FileText } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { ReportsModal } from "@/components/lists/lab/ReportsModal";
import { api } from "@/lib/api";
import type { ArtifactJob } from "@/lib/types";

interface ToolSectionProps {
  listId: string;
  selectedSourceIds: string[];
}

const TOOL_BTN =
  "flex flex-col items-center gap-1 rounded-xl border border-border bg-white/64 px-2 py-3 text-center transition hover:bg-white hover:shadow-sm disabled:cursor-not-allowed disabled:opacity-40";

export function ToolSection({ listId, selectedSourceIds }: ToolSectionProps) {
  const { t } = useLanguage();
  const { trackJobs } = useJobActivity();
  const [reportsOpen, setReportsOpen] = useState(false);
  const [mindMapPending, setMindMapPending] = useState(false);

  const hasSelection = selectedSourceIds.length > 0;

  async function handleMindMap() {
    if (!hasSelection || mindMapPending) return;
    setMindMapPending(true);
    try {
      // Mind-map jobs use a fixed directive on the worker side; the
      // placeholder prompt here just satisfies the API contract — the
      // backend rebinds before the LLM sees anything.
      const job = (await api.createArtifact(listId, {
        type: "mind_map",
        prompt: "Generate a hierarchical mind map of the selected sources.",
        source_ids: selectedSourceIds,
      })) as ArtifactJob;
      trackJobs([{ id: job.id, producer: "artifact", label: "mind_map", contextKey: listId }]);
    } finally {
      setMindMapPending(false);
    }
  }

  return (
    <div
      className="grid shrink-0 grid-cols-2 gap-2 px-4 py-3 sm:grid-cols-3 lg:grid-cols-4"
      data-testid="tool-section"
    >
      {[
        {
          label: t("lab.toolSection.reports"),
          icon: FileText,
          onClick: () => hasSelection && setReportsOpen(true),
          disabled: !hasSelection,
        },
        {
          label: t("lab.toolSection.mindMap"),
          icon: Brain,
          onClick: handleMindMap,
          disabled: !hasSelection || mindMapPending,
        },
      ].map(({ label, icon: Icon, onClick, disabled }) => (
        <button
          key={label}
          type="button"
          onClick={onClick}
          disabled={disabled}
          className={TOOL_BTN}
        >
          <Icon size={16} className="text-muted" />
          <span className="text-xs font-medium text-ink">{label}</span>
        </button>
      ))}

      <ReportsModal
        open={reportsOpen}
        listId={listId}
        sourceIds={selectedSourceIds}
        onClose={() => setReportsOpen(false)}
      />
    </div>
  );
}
