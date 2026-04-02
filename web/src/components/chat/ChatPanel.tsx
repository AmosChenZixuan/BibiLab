import { mutedTextClass, workspacePanelBodyClass, workspacePanelTitleClass } from "../../lib/ui";
import { Panel } from "../../components/ui";

export function ChatPanel() {
  return (
    <Panel variant="workspace">
      <h2 className={workspacePanelTitleClass}>Chat</h2>
      <div className={`${workspacePanelBodyClass} min-h-[490px] content-center`}>
        <div className="grid gap-2.5">
          <div className="h-3.5 w-[86%] rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
          <div className="h-3.5 rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
          <div className="h-3.5 w-[54%] rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
        </div>
        <p className={mutedTextClass}>List-scoped chat arrives in v1. This panel stays intentionally quiet until then.</p>
      </div>
    </Panel>
  );
}
