import { ReactNode, useEffect, useId } from "react";
import { createPortal } from "react-dom";

type ModalProps = {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
  size?: "md" | "lg" | "xl";
};

const sizes = {
  md: "max-w-xl",
  lg: "max-w-3xl",
  xl: "max-w-5xl",
};

export function Modal({ open, onClose, title, children, footer, size = "md" }: ModalProps) {
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
      className="fixed inset-0 z-modal flex items-center justify-center bg-scrim px-4 py-8 backdrop-blur-sm"
      data-testid="modal-backdrop"
      onClick={(event) => event.stopPropagation()}
      onMouseUp={(event) => {
        event.stopPropagation();
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        aria-labelledby={titleId}
        aria-modal="true"
        className={`w-full ${sizes[size]} min-h-100 rounded-3xl border border-white/60 bg-white/95 shadow-lg`}
        role="dialog"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="px-8 pt-8">
          <h2 className="m-0 text-base font-semibold tracking-tight text-ink" id={titleId}>
            {title}
          </h2>
        </div>
        <div className="grid gap-5 px-8 py-6">{children}</div>
        {footer ? <div className="flex justify-end gap-2 px-8 pb-8">{footer}</div> : null}
      </div>
    </div>,
    document.body,
  );
}
