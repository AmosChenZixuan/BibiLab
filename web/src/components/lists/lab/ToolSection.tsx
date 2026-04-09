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
    <div className="grid grid-cols-2 gap-3 p-4" data-testid="tool-section">
      <button
        type="button"
        onClick={() => setReportsOpen(true)}
        className="flex flex-col items-center gap-2 rounded-2xl border border-border bg-white/64 px-4 py-4 text-center transition hover:bg-white hover:shadow-sm"
      >
        <FileText size={20} className="text-muted" />
        <span className="text-sm font-medium text-ink">Reports</span>
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
