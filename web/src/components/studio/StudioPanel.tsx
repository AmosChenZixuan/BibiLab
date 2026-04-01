type Props = {
  busy: boolean;
  error: string | null;
  status: string | null;
  onGenerate: () => Promise<void>;
};

export function StudioPanel({ busy, error, status, onGenerate }: Props) {
  return (
    <section className="workspace-panel">
      <h2 className="workspace-panel__title">Studio</h2>
      <div className="workspace-panel__body">
        <p className="page-lede">
          Generate a list-level markdown overview from the sources already processed into this notebook.
        </p>
        <button className="primary-button" disabled={busy} onClick={() => void onGenerate()} type="button">
          {busy ? "Generating..." : "Generate overview"}
        </button>
        {status ? <p className="status-message success">{status}</p> : null}
        {error ? <p className="status-message error">{error}</p> : null}
      </div>
    </section>
  );
}
