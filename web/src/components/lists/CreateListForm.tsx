import { useState } from "react";

import { Button, FormField, Input, Panel } from "../../components/ui";

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
          <h2 className="m-0 font-serif text-2xl">Start a list</h2>
          <p className="m-0 text-muted">Create a destination before you queue any source URLs.</p>
        </div>
      </div>
      <form className="mt-4 grid gap-4" onSubmit={handleSubmit}>
        <FormField label="List name">
          <Input
            onChange={(event) => setName(event.target.value)}
            placeholder="Systems, Research, History of Film"
            value={name}
          />
        </FormField>
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="primary" disabled={busy} type="submit">
            {busy ? "Creating..." : "Create list"}
          </Button>
          {error ? <p className="m-0 text-sm text-danger">{error}</p> : null}
        </div>
      </form>
    </Panel>
  );
}
