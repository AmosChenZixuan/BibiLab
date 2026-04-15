import { useState } from "react";
import { FileText } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { ReportsModal } from "@/components/lists/lab/ReportsModal";
import type { ArtifactType } from "@/lib/types";

interface ToolSectionProps {
  listId: string;
  selectedSourceIds: string[];
  onArtifactGenerated: (artifactId: string, type: ArtifactType, sourceIds: string[]) => void;
}

export function ToolSection({ listId, selectedSourceIds, onArtifactGenerated }: ToolSectionProps) {
  const { t } = useLanguage();
  const [reportsOpen, setReportsOpen] = useState(false);

  return (
    <div className="grid shrink-0 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 px-4 py-3" data-testid="tool-section">
      <button
        type="button"
        onClick={() => selectedSourceIds.length > 0 && setReportsOpen(true)}
        disabled={selectedSourceIds.length === 0}
        className="flex flex-col items-center gap-1 rounded-xl border border-border bg-white/64 px-2 py-3 text-center transition hover:bg-white hover:shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
      >
        <FileText size={16} className="text-muted" />
        <span className="text-xs font-medium text-ink">{t("lab.toolSection.reports")}</span>
      </button>

      <ReportsModal
        open={reportsOpen}
        listId={listId}
        sourceIds={selectedSourceIds}
        onClose={() => setReportsOpen(false)}
        onArtifactGenerated={onArtifactGenerated}
      />
    </div>
  );
}
