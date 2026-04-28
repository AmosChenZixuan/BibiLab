import { useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { CHAT_MODE_FOCUSED, CHAT_MODE_BROAD, type ChatMode } from "@/lib/constants";
import { api, toErrorMessageWithT } from "@/lib/api";
import { Modal } from "@/components/ui/Modal";

interface ChatConfigModalProps {
  listId: string;
  currentMode: ChatMode;
  onClose: () => void;
  onSave: (mode: ChatMode) => void;
}

export function ChatConfigModal({ listId, currentMode, onClose, onSave }: ChatConfigModalProps) {
  const { t } = useLanguage();
  const [selected, setSelected] = useState<ChatMode>((currentMode as ChatMode) || CHAT_MODE_FOCUSED);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      await api.updateConversation(listId, { mode: selected });
      onSave(selected);
      onClose();
    } catch (err) {
      setError(toErrorMessageWithT(err, t));
      console.error("[ChatConfigModal] save failed:", err);
      setSaving(false);
    }
  }

  return (
    <Modal
      open={true}
      onClose={onClose}
      title={t("chat.configModal.title")}
      size="md"
      footer={
        <>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full px-3 py-1.5 text-xs text-muted transition hover:bg-border"
          >
            {t("chat.configModal.cancel")}
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={saving}
            className="rounded-full bg-ink px-3 py-1.5 text-xs font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {saving ? t("chat.configModal.saving") : t("chat.configModal.save")}
          </button>
        </>
      }
    >
      <div
        className="flex gap-0.5 rounded-full bg-surface p-1"
        role="radiogroup"
        aria-label={t("chat.configModal.label")}
      >
        {([CHAT_MODE_FOCUSED, CHAT_MODE_BROAD] as const).map((mode) => {
          const isSelected = selected === mode;
          return (
            <label
              key={mode}
              className={`flex flex-1 cursor-pointer flex-col items-center rounded-full px-3 py-2 text-center text-sm transition ${
                isSelected
                  ? "bg-blue text-white shadow-sm"
                  : "text-muted hover:text-ink"
              }`}
            >
              <input
                type="radio"
                name="chat-mode"
                value={mode}
                checked={isSelected}
                onChange={() => setSelected(mode)}
                className="sr-only"
              />
              <span className="font-medium">{t(`chat.configModal.${mode}`)}</span>
              <span className={`text-xs ${isSelected ? "text-white/80" : "text-muted"}`}>
                {t(`chat.configModal.${mode}Hint`)}
              </span>
            </label>
          );
        })}
      </div>

      {error && (
        <p className="mt-2 text-xs text-pink">{error}</p>
      )}
    </Modal>
  );
}
