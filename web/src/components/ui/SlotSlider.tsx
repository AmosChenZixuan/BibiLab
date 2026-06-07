import { useId, useRef } from "react";

/**
 * 4-position segmented control. Renders a radiogroup with one radio per slot;
 * arrow keys move focus, Space/Enter selects, Home/End jump to ends.
 *
 * Visual treatment is intentionally minimal in this commit — color, spacing,
 * and hover/focus rings are TBD pending the slot-slider visual review. The
 * data + behavior contract is locked: parent owns state, onChange fires on
 * every selection, aria-label flows from the surrounding SettingsField hint.
 */

export interface SlotOption<T extends string | number> {
  value: T;
  label: string;
}

interface Props<T extends string | number> {
  options: SlotOption<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
  id?: string;
  className?: string;
  disabled?: boolean;
}

function clsx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter((p): p is string => Boolean(p)).join(" ");
}

export function SlotSlider<T extends string | number>({
  options,
  value,
  onChange,
  ariaLabel,
  id,
  className,
  disabled = false,
}: Props<T>) {
  const generatedId = useId();
  const groupId = id ?? generatedId;

  // Roving tabindex: keyboard selection must also move DOM focus to the newly
  // selected radio, otherwise the focus ring is stranded on the old (now
  // tabIndex=-1) button while a different one is aria-checked.
  const buttonRefs = useRef(new Map<string, HTMLButtonElement>());

  function selectAndFocus(next: T) {
    onChange(next);
    buttonRefs.current.get(String(next))?.focus();
  }

  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      id={groupId}
      className={clsx("inline-flex w-full rounded-xl border border-border bg-white/60 p-1", className)}
    >
      {options.map((option) => {
        const checked = option.value === value;
        return (
          <button
            key={String(option.value)}
            ref={(el) => {
              if (el) buttonRefs.current.set(String(option.value), el);
              else buttonRefs.current.delete(String(option.value));
            }}
            type="button"
            role="radio"
            aria-checked={checked}
            aria-label={option.label}
            tabIndex={checked ? 0 : -1}
            disabled={disabled}
            onClick={() => {
              if (!checked) onChange(option.value);
            }}
            onKeyDown={(event) => {
              if (disabled) return;
              if (event.key === "ArrowRight" || event.key === "ArrowDown") {
                event.preventDefault();
                const idx = options.findIndex((o) => o.value === value);
                selectAndFocus(options[(idx + 1) % options.length].value);
              } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                event.preventDefault();
                const idx = options.findIndex((o) => o.value === value);
                selectAndFocus(options[(idx - 1 + options.length) % options.length].value);
              } else if (event.key === "Home") {
                event.preventDefault();
                selectAndFocus(options[0].value);
              } else if (event.key === "End") {
                event.preventDefault();
                selectAndFocus(options[options.length - 1].value);
              } else if (event.key === " " || event.key === "Enter") {
                event.preventDefault();
                if (!checked) onChange(option.value);
              }
            }}
            className={clsx(
              "flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition",
              "outline-none focus-visible:ring-2 focus-visible:ring-sky/45",
              checked ? "bg-blue/15 text-blue" : "text-muted hover:text-ink hover:bg-white/60",
              disabled && "cursor-not-allowed opacity-60",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
