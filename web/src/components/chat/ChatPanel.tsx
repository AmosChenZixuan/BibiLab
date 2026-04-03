import { Panel, PanelBody, PanelTitle } from "../../components/ui";

export function ChatPanel() {
  return (
    <Panel variant="workspace">
      <PanelTitle>Chat</PanelTitle>
      <PanelBody className="min-h-96 content-center">
        <div className="grid gap-2.5">
          <div className="h-3.5 w-5/6 rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
          <div className="h-3.5 rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
          <div className="h-3.5 w-1/2 rounded-full bg-linear-to-r from-pink/14 to-sky/14" />
        </div>
        <p className="m-0 text-muted">List-scoped chat arrives in v1. This panel stays intentionally quiet until then.</p>
      </PanelBody>
    </Panel>
  );
}
