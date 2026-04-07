import { useState } from "react";

import type { BibilabList } from "@/lib/types";
import { useLanguage } from "@/app/LanguageContext";
import { Button, Input, Modal } from "@/components/ui";

interface RenameListModalProps {
  list: BibilabList | null;
  open: boolean;
  onClose: () => void;
  onCommit: (newName: string) => Promise<void>;
}

export function RenameListModal({ list, open, onClose, onCommit }: RenameListModalProps) {
  const { t } = useLanguage();
  const [draft, setDraft] = useState("");

  function handleClose() {
    setDraft("");
    onClose();
  }

  async function handleCommit() {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === list?.name) {
      handleClose();
      return;
    }
    await onCommit(trimmed);
    handleClose();
  }

  return (
    <Modal
      footer={
        <>
          <Button onClick={handleClose} size="sm" variant="ghost">
            {t("common.cancel")}
          </Button>
          <Button onClick={() => void handleCommit()} size="sm" variant="primary">
            {t("common.save")}
          </Button>
        </>
      }
      onClose={handleClose}
      open={open}
      size="lg"
      title={t("home.renameList")}
    >
      <div className="relative h-100 overflow-hidden rounded-3xl bg-pink-100 shadow-lg">
        {list?.thumbnail_url ? (
          <div
            className="absolute inset-0 bg-cover bg-center"
            style={{ backgroundImage: `url("${list.thumbnail_url}")` }}
          />
        ) : null}
        <div className="absolute inset-0 bg-linear-to-t from-black/65 via-black/20 to-transparent" />
        <div className="absolute inset-x-6 bottom-6 z-10">
          <span className="block text-3xl font-semibold tracking-tighter leading-tight text-white">
            {draft || list?.name || t("home.untitledList")}
          </span>
        </div>
      </div>
      <div className="">
        <label className="grid gap-2">
          <span className="text-xs font-semibold uppercase tracking-widest text-muted">{t("home.listName")}</span>
          <Input
            aria-label={t("home.listName")}
            autoFocus
            className="select-text rounded-2xl bg-white/92 px-4 py-3 text-2xl leading-tight font-normal tracking-normal text-ink focus:border-blue/25 focus:ring-2 focus:ring-sky/10"
            placeholder={t("home.untitledList")}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void handleCommit();
              }
            }}
            value={draft}
          />
        </label>
      </div>
    </Modal>
  );
}
