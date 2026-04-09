import { useState } from "react";
import { ArrowLeftToLine, ArrowRightToLine, Minimize2 } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { COLLAPSED_PANEL, MIN_PANEL } from "@/components/lists/panel-resize";
import type { Artifact } from "@/lib/types";

import { ArtifactList } from "./lab/ArtifactList";
import { ArtifactViewer } from "./lab/ArtifactViewer";
import { ToolSection } from "./lab/ToolSection";

type LabMode = "tool-list" | "viewer" | "collapsed";

interface LabPanelProps {
  listId: string;
  labCollapsed: boolean;
  labW: number;
  sourceIds: string[];
  onToggleCollapse: () => void;
}

export function LabPanel({ listId, labCollapsed, labW, sourceIds, onToggleCollapse }: LabPanelProps) {
  const { t } = useLanguage();
  const panelBase = "flex h-full shrink-0 flex-col overflow-hidden";

  const [labMode, setLabMode] = useState<LabMode>("tool-list");
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);

  function handleViewArtifact(artifact: Artifact) {
    setSelectedArtifact(artifact);
    setLabMode("viewer");
  }

  function getHeaderButton() {
    if (labCollapsed) {
      return {
        label: "expand",
        icon: <ArrowLeftToLine size={16} />,
        onClick: onToggleCollapse,
      };
    }
    if (labMode === "tool-list") {
      return {
        label: "collapse",
        icon: <ArrowRightToLine size={16} />,
        onClick: onToggleCollapse,
      };
    }
    return {
      label: "minimize",
      icon: <Minimize2 size={16} />,
      onClick: () => setLabMode("tool-list"),
    };
  }

  const headerBtn = getHeaderButton();

  return (
    <div
      style={
        labCollapsed
          ? { width: `${COLLAPSED_PANEL}px`, minWidth: `${COLLAPSED_PANEL}px` }
          : { width: `${labW}px`, minWidth: `${MIN_PANEL}px` }
      }
      className={panelBase}
    >
      <div className="flex shrink-0 items-center border-b border-border px-4 py-4">
        {!labCollapsed && (
          <h2 className="m-0 flex-1 font-serif text-lg text-ink">{t("lists.lab")}</h2>
        )}
        <button
          type="button"
          onClick={headerBtn.onClick}
          aria-label={headerBtn.label}
          className={`flex h-7 w-7 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink ${labCollapsed ? "mx-auto" : ""}`}
        >
          {headerBtn.icon}
        </button>
      </div>

      {!labCollapsed && (
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {labMode === "tool-list" ? (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <ToolSection listId={listId} sourceIds={sourceIds} />
              <div className="flex min-h-0 flex-1 flex-col border-t border-border px-4 pt-4">
                <ArtifactList listId={listId} onViewArtifact={handleViewArtifact} />
              </div>
            </div>
          ) : (
            selectedArtifact && <ArtifactViewer artifact={selectedArtifact} />
          )}
        </div>
      )}
    </div>
  );
}
