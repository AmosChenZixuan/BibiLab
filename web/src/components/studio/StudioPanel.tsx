import { mutedTextClass, statusErrorClass, statusSuccessClass, workspacePanelBodyClass, workspacePanelTitleClass } from "../../lib/ui";
import { Button, Panel } from "../../components/ui";

type Props = {
  busy: boolean;
  error: string | null;
  status: string | null;
  onGenerate: () => Promise<void>;
};

export function StudioPanel({ busy, error, status, onGenerate }: Props) {
  return (
    <Panel variant="workspace">
      <h2 className={workspacePanelTitleClass}>Studio</h2>
      <div className={workspacePanelBodyClass}>
        <p className={mutedTextClass}>
          Generate a list-level markdown overview from the sources already processed into this notebook.
        </p>
        <Button variant="primary" disabled={busy} onClick={() => void onGenerate()} type="button">
          {busy ? "Generating..." : "Generate overview"}
        </Button>
        {status ? <p className={statusSuccessClass}>{status}</p> : null}
        {error ? <p className={statusErrorClass}>{error}</p> : null}
      </div>
    </Panel>
  );
}
