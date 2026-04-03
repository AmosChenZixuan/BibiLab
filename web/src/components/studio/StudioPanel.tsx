import { Button, Panel, PanelBody, PanelTitle } from "../../components/ui";

type Props = {
  busy: boolean;
  error: string | null;
  status: string | null;
  onGenerate: () => Promise<void>;
};

export function StudioPanel({ busy, error, status, onGenerate }: Props) {
  return (
    <Panel variant="workspace">
      <PanelTitle>Studio</PanelTitle>
      <PanelBody>
        <p className="m-0 text-muted">
          Generate a list-level markdown overview from the sources already processed into this notebook.
        </p>
        <Button variant="primary" disabled={busy} onClick={() => void onGenerate()} type="button">
          {busy ? "Generating..." : "Generate overview"}
        </Button>
        {status ? <p className="m-0 text-sm text-sky-600">{status}</p> : null}
        {error ? <p className="m-0 text-sm text-rose-900">{error}</p> : null}
      </PanelBody>
    </Panel>
  );
}
