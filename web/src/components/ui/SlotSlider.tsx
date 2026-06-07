import { useId } from "react";

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
                const next = options[(idx + 1) % options.length];
                onChange(next.value);
              } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                event.preventDefault();
                const idx = options.findIndex((o) => o.value === value);
                const prev = options[(idx - 1 + options.length) % options.length];
                onChange(prev.value);
              } else if (event.key === "Home") {
                event.preventDefault();
                onChange(options[0].value);
              } else if (event.key === "End") {
                event.preventDefault();
                onChange(options[options.length - 1].value);
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
