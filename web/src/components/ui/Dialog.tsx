import { ReactNode, useEffect, useId } from "react";
import { createPortal } from "react-dom";

type DialogProps = {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
};

export function Dialog({ open, onClose, title, children, footer }: DialogProps) {
  const titleId = useId();

  useEffect(() => {
    if (!open) {
      return;
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose, open]);

  if (!open) {
    return null;
  }

  return createPortal(
    <div
      className="fixed inset-0 z-overlay flex items-center justify-center bg-black/30 px-4 backdrop-blur-[4px]"
      data-testid="dialog-backdrop"
      onClick={onClose}
    >
      <div
        aria-labelledby={titleId}
        aria-modal="true"
        className="w-full max-w-[360px] rounded-[18px] border border-border bg-white shadow-elevated"
        role="dialog"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 pt-5">
          <h2 className="m-0 text-[15px] font-semibold tracking-[-0.02em] text-ink" id={titleId}>
            {title}
          </h2>
          <button
            aria-label="Close dialog"
            className="inline-flex h-[26px] w-[26px] items-center justify-center rounded-full border-0 bg-black/5 text-muted transition hover:bg-black/10"
            onClick={onClose}
            type="button"
          >
            x
          </button>
        </div>
        <div className="grid gap-3 px-5 py-4">{children}</div>
        {footer ? <div className="flex justify-end gap-2 px-5 pb-5">{footer}</div> : null}
      </div>
    </div>,
    document.body,
  );
}
