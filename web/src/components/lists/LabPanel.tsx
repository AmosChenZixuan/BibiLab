import { useState } from "react";
import { ChevronLeft, ChevronRight, Minimize2 } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { COLLAPSED_PANEL, MIN_PANEL } from "@/components/lists/panel-resize";

type LabMode = "tool-list" | "viewer" | "collapsed";

interface LabPanelProps {
  listId: string;
  labCollapsed: boolean;
  labW: number;
  onToggleCollapse: () => void;
}

export function LabPanel({ listId, labCollapsed, labW, onToggleCollapse }: LabPanelProps) {
  const { t } = useLanguage();
  const panelBase = "flex shrink-0 flex-col overflow-hidden rounded-3xl border border-border bg-white/76 shadow-lg";

  const [labMode, setLabMode] = useState<LabMode>("tool-list");

  function getHeaderButton() {
    if (labCollapsed) {
      return {
        label: "expand",
        icon: <ChevronRight size={16} />,
        onClick: onToggleCollapse,
      };
    }
    if (labMode === "tool-list") {
      return {
        label: "collapse",
        icon: <ChevronLeft size={16} />,
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
          {labMode === "tool-list" && (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <div className="border-b border-border px-4 py-3">
                <p className="m-0 text-sm text-muted">Tool</p>
              </div>
              <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
                <button
                  type="button"
                  onClick={() => setLabMode("viewer")}
                  className="m-0 flex h-full items-center justify-center px-4 py-3 text-sm text-muted transition hover:bg-border hover:text-ink"
                >
                  Artifact
                </button>
              </div>
            </div>
          )}
          {labMode === "viewer" && (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <p className="m-0 px-4 py-3 text-sm text-muted">Artifact content</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
