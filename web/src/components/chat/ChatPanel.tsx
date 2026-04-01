import { mutedTextClass, workspacePanelBodyClass, workspacePanelClass, workspacePanelTitleClass } from "../../lib/ui";

export function ChatPanel() {
  return (
    <section className={workspacePanelClass}>
      <h2 className={workspacePanelTitleClass}>Chat</h2>
      <div className={`${workspacePanelBodyClass} min-h-[490px] content-center`}>
        <div className="grid gap-2.5">
          <div className="h-[14px] w-[86%] rounded-full bg-[linear-gradient(90deg,rgba(240,139,185,0.14),rgba(125,217,255,0.14))]" />
          <div className="h-[14px] rounded-full bg-[linear-gradient(90deg,rgba(240,139,185,0.14),rgba(125,217,255,0.14))]" />
          <div className="h-[14px] w-[54%] rounded-full bg-[linear-gradient(90deg,rgba(240,139,185,0.14),rgba(125,217,255,0.14))]" />
        </div>
        <p className={mutedTextClass}>List-scoped chat arrives in v1. This panel stays intentionally quiet until then.</p>
      </div>
    </section>
  );
}
