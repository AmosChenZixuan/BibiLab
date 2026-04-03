import { ReactNode, useEffect, useId, useRef } from "react";
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
  const backdropPressStarted = useRef(false);

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
      onClick={(event) => {
        event.stopPropagation();
        if (backdropPressStarted.current && event.target === event.currentTarget) {
          onClose();
        }
        backdropPressStarted.current = false;
      }}
      onMouseDown={(event) => {
        event.stopPropagation();
        backdropPressStarted.current = event.target === event.currentTarget;
      }}
    >
      <div
        aria-labelledby={titleId}
        aria-modal="true"
        className={`w-full ${sizes[size]} rounded-drawer border border-white/60 bg-surface-strong shadow-elevated`}
        role="dialog"
        onClick={(event) => event.stopPropagation()}
        onMouseDown={() => {
          backdropPressStarted.current = false;
        }}
      >
        <div className="px-8 pt-8">
          <h2 className="m-0 text-[15px] font-semibold tracking-[-0.02em] text-ink" id={titleId}>
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
