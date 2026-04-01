import { useState } from "react";
import {
  appPanelClass,
  fieldClass,
  fieldLabelClass,
  inputClass,
  primaryButtonClass,
  mutedTextClass,
  sectionTitleClass,
  statusErrorClass,
} from "../../lib/ui";

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
    <section className={appPanelClass}>
      <div className="flex flex-wrap items-center gap-3">
        <div>
          <h2 className={sectionTitleClass}>Start a list</h2>
          <p className={mutedTextClass}>Create a destination before you queue any source URLs.</p>
        </div>
      </div>
      <form className="mt-4 grid gap-4" onSubmit={handleSubmit}>
        <label className={fieldClass}>
          <span className={fieldLabelClass}>List name</span>
          <input
            aria-label="List name"
            className={inputClass}
            onChange={(event) => setName(event.target.value)}
            placeholder="Systems, Research, History of Film"
            value={name}
          />
        </label>
        <div className="flex flex-wrap items-center gap-3">
          <button className={primaryButtonClass} disabled={busy} type="submit">
            {busy ? "Creating..." : "Create list"}
          </button>
          {error ? <p className={statusErrorClass}>{error}</p> : null}
        </div>
      </form>
    </section>
  );
}
