import { ReactNode, Ref, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

type ContextMenuItem = {
  label: string;
  icon?: ReactNode;
  onClick: () => void;
  variant?: "danger";
};

type TriggerArgs = {
  open: boolean;
  toggle: () => void;
  triggerRef: Ref<HTMLButtonElement>;
};

type ContextMenuProps = {
  items: ContextMenuItem[];
  trigger: (args: TriggerArgs) => ReactNode;
};

type Position = {
  top: number;
  left: number;
};

let openMenuId: string | null = null;
const menuListeners = new Map<string, () => void>();

export function ContextMenu({ items, trigger }: ContextMenuProps) {
  const menuId = useId();
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<Position>({ top: 0, left: 0 });

  useLayoutEffect(() => {
    if (!open || !triggerRef.current) {
      return;
    }

    const rect = triggerRef.current.getBoundingClientRect();
    setPosition({
      top: rect.bottom + 8,
      left: Math.max(8, rect.right - 168),
    });
  }, [open]);

  useEffect(() => {
    if (!open) {
      menuListeners.delete(menuId);
      if (openMenuId === menuId) {
        openMenuId = null;
      }
      return;
    }

    const closeMenu = () => setOpen(false);
    menuListeners.set(menuId, closeMenu);

    function handlePointerDown(event: MouseEvent) {
      const target = event.target as Node | null;
      if (
        target &&
        !menuRef.current?.contains(target) &&
        !triggerRef.current?.contains(target)
      ) {
        closeMenu();
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        closeMenu();
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
      menuListeners.delete(menuId);
      if (openMenuId === menuId) {
        openMenuId = null;
      }
    };
  }, [menuId, open]);

  function toggle() {
    if (open) {
      setOpen(false);
      return;
    }

    if (openMenuId && openMenuId !== menuId) {
      menuListeners.get(openMenuId)?.();
    }

    openMenuId = menuId;
    setOpen(true);
  }

  return (
    <>
      {trigger({ open, toggle, triggerRef })}
      {open
        ? createPortal(
            <div
              className="fixed z-float min-w-40 rounded-lg border border-border bg-white p-1 shadow-lg"
              ref={menuRef}
              role="menu"
              style={{ top: `${position.top}px`, left: `${position.left}px` }}
            >
              {items.map((item) => (
                <button
                  className={`flex w-full items-center gap-2 rounded-md border-0 px-2.5 py-2 text-left text-sm font-medium transition hover:bg-black/5 ${
                    item.variant === "danger" ? "text-rose-900" : "text-ink"
                  }`}
                  key={item.label}
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    item.onClick();
                    setOpen(false);
                  }}
                  role="menuitem"
                  type="button"
                >
                  {item.icon ? <span className="inline-flex size-4.5 items-center justify-center">{item.icon}</span> : null}
                  <span>{item.label}</span>
                </button>
              ))}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
