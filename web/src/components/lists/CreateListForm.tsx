import { useState } from "react";

import {
  inputClass,
  mutedTextClass,
  sectionTitleClass,
  statusErrorClass,
} from "../../lib/ui";
import { Button, FormField, Panel } from "../../components/ui";

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
    <Panel variant="app">
      <div className="flex flex-wrap items-center gap-3">
        <div>
          <h2 className={sectionTitleClass}>Start a list</h2>
          <p className={mutedTextClass}>Create a destination before you queue any source URLs.</p>
        </div>
      </div>
      <form className="mt-4 grid gap-4" onSubmit={handleSubmit}>
        <FormField label="List name">
          <input
            className={inputClass}
            onChange={(event) => setName(event.target.value)}
            placeholder="Systems, Research, History of Film"
            value={name}
          />
        </FormField>
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="primary" disabled={busy} type="submit">
            {busy ? "Creating..." : "Create list"}
          </Button>
          {error ? <p className={statusErrorClass}>{error}</p> : null}
        </div>
      </form>
    </Panel>
  );
}
