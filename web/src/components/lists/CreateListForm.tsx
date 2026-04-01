import { useState } from "react";

type Props = {
  busy: boolean;
  error: string | null;
  onCreate: (name: string) => Promise<void>;
};

export function CreateListForm({ busy, error, onCreate }: Props) {
  const [name, setName] = useState("");

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) {
      return;
    }
    await onCreate(trimmed);
    setName("");
  }

  return (
    <section className="panel">
      <div className="row">
        <div>
          <h2 className="list-card__title">Start a list</h2>
          <p className="page-lede">Create a destination before you queue any source URLs.</p>
        </div>
      </div>
      <form className="form-stack" onSubmit={handleSubmit}>
        <label className="field">
          <span>List name</span>
          <input
            aria-label="List name"
            onChange={(event) => setName(event.target.value)}
            placeholder="Systems, Research, History of Film"
            value={name}
          />
        </label>
        <div className="inline-actions">
          <button className="primary-button" disabled={busy} type="submit">
            {busy ? "Creating..." : "Create list"}
          </button>
          {error ? <p className="status-message error">{error}</p> : null}
        </div>
      </form>
    </section>
  );
}
