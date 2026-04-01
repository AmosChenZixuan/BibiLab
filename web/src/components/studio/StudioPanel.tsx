import { mutedTextClass, primaryButtonClass, statusErrorClass, statusSuccessClass, workspacePanelBodyClass, workspacePanelClass, workspacePanelTitleClass } from "../../lib/ui";

type Props = {
  busy: boolean;
  error: string | null;
  status: string | null;
  onGenerate: () => Promise<void>;
};

export function StudioPanel({ busy, error, status, onGenerate }: Props) {
  return (
    <section className={workspacePanelClass}>
      <h2 className={workspacePanelTitleClass}>Studio</h2>
      <div className={workspacePanelBodyClass}>
        <p className={mutedTextClass}>
          Generate a list-level markdown overview from the sources already processed into this notebook.
        </p>
        <button className={primaryButtonClass} disabled={busy} onClick={() => void onGenerate()} type="button">
          {busy ? "Generating..." : "Generate overview"}
        </button>
        {status ? <p className={statusSuccessClass}>{status}</p> : null}
        {error ? <p className={statusErrorClass}>{error}</p> : null}
      </div>
    </section>
  );
}
