import { Copy } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";

interface ViewPromptModalProps {
  open: boolean;
  onClose: () => void;
  prompt: string;
}

export function ViewPromptModal({ open, onClose, prompt }: ViewPromptModalProps) {
  const { t } = useLanguage();
  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(prompt);
    } catch {
      // Clipboard API not available
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t("lab.viewPromptModal.title")}
      size="lg"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            {t("lab.viewPromptModal.close")}
          </Button>
          <Button onClick={handleCopy}>
            <Copy size={14} />
            {t("lab.viewPromptModal.copy")}
          </Button>
        </>
      }
    >
      <pre className="whitespace-pre-wrap rounded-lg border border-divider bg-white/64 p-4 text-caption text-charcoal">
        {prompt}
      </pre>
    </Modal>
  );
}
