import { useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { Button, FormField, Input, Panel } from "@/components/ui";

type Props = {
  busy: boolean;
  error: string | null;
  onCreate: (name: string) => Promise<void>;
};

export function CreateListForm({ busy, error, onCreate }: Props) {
  const { t } = useLanguage();
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
          <h2 className="m-0 font-serif text-2xl">{t("home.startList")}</h2>
          <p className="m-0 text-muted">{t("home.startListDesc")}</p>
        </div>
      </div>
      <form className="mt-4 grid gap-4" onSubmit={handleSubmit}>
        <FormField label={t("home.listName")}>
          <Input
            onChange={(event) => setName(event.target.value)}
            placeholder={t("home.listPlaceholder")}
            value={name}
          />
        </FormField>
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="primary" disabled={busy} type="submit">
            {busy ? t("common.creating") : t("common.createList")}
          </Button>
          {error ? <p className="m-0 text-sm text-rose-900">{error}</p> : null}
        </div>
      </form>
    </Panel>
  );
}
