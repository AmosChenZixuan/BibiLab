import { ReactNode, useCallback, useId, useRef, useEffect } from "react";
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

const FOCUSABLE_SELECTORS = [
  "button:not([disabled])",
  "[href]",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

export function Modal({ open, onClose, title, children, footer, size = "md" }: ModalProps) {
  const titleId = useId();
  const backdropPressStarted = useRef(false);
  const previousActiveElement = useRef<HTMLElement | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  const getFocusableElements = useCallback((container: HTMLElement) => {
    return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
  }, []);

  // Handle ESC key and focus trap
  useEffect(() => {
    if (!open) {
      return;
    }

    // Store the previously focused element and lock body scroll
    previousActiveElement.current = document.activeElement as HTMLElement;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    // Focus the dialog
    const dialog = dialogRef.current;
    if (dialog) {
      const focusable = getFocusableElements(dialog);
      if (focusable.length > 0) {
        focusable[0].focus();
      } else {
        dialog.focus();
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const dialog = dialogRef.current;
      if (!dialog) {
        return;
      }

      const focusable = getFocusableElements(dialog);
      if (focusable.length === 0) {
        event.preventDefault();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (event.shiftKey) {
        if (document.activeElement === first) {
          event.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      // Restore body scroll
      document.body.style.overflow = originalOverflow;
      // Restore focus to the element that opened the modal
      if (previousActiveElement.current && previousActiveElement.current.focus) {
        previousActiveElement.current.focus();
      }
    };
  }, [open, onClose, getFocusableElements]);

  if (!open) {
    return null;
  }

  return createPortal(
    <div
      className="fixed inset-0 z-modal flex items-center justify-center bg-overlay px-4 py-8 backdrop-blur-sm"
      data-testid="modal-backdrop"
      onClick={(event) => event.stopPropagation()}
      onMouseUp={(event) => {
        event.stopPropagation();
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
      onMouseDown={(event) => {
        event.stopPropagation();
        backdropPressStarted.current = event.target === event.currentTarget;
      }}
    >
      <div
        ref={dialogRef}
        aria-labelledby={titleId}
        aria-modal="true"
        className={`w-full ${sizes[size]} min-h-100 rounded-3xl border border-white/60 bg-white/95 shadow-lg`}
        role="dialog"
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
        onMouseDown={() => {
          backdropPressStarted.current = false;
        }}
      >
        <div className="px-8 pt-8">
          <h2 className="m-0 text-base font-semibold tracking-tight text-charcoal" id={titleId}>
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
