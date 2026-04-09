import { Copy } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";

interface ViewPromptModalProps {
  open: boolean;
  onClose: () => void;
  prompt: string;
}

export function ViewPromptModal({ open, onClose, prompt }: ViewPromptModalProps) {
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
      title="Prompt"
      size="lg"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
          <Button onClick={handleCopy}>
            <Copy size={14} />
            Copy
          </Button>
        </>
      }
    >
      <pre className="whitespace-pre-wrap rounded-lg border border-border bg-white/64 p-4 text-sm text-ink">
        {prompt}
      </pre>
    </Modal>
  );
}
