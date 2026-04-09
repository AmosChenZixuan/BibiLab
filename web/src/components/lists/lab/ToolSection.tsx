import { useState } from "react";
import { FileText } from "lucide-react";

import { ReportsModal } from "@/components/lists/lab/ReportsModal";

interface ToolSectionProps {
  listId: string;
  sourceIds: string[];
}

export function ToolSection({ listId, sourceIds }: ToolSectionProps) {
  const [reportsOpen, setReportsOpen] = useState(false);

  return (
    <div className="grid shrink-0 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 px-4 py-3" data-testid="tool-section">
      <button
        type="button"
        onClick={() => setReportsOpen(true)}
        className="flex flex-col items-center gap-1 rounded-xl border border-border bg-white/64 px-2 py-3 text-center transition hover:bg-white hover:shadow-sm"
      >
        <FileText size={16} className="text-muted" />
        <span className="text-xs font-medium text-ink">Reports</span>
      </button>

      <ReportsModal
        open={reportsOpen}
        listId={listId}
        sourceIds={sourceIds}
        onClose={() => setReportsOpen(false)}
      />
    </div>
  );
}
