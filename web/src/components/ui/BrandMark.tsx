import type { ComponentPropsWithoutRef } from "react";

// The implementation overrides these slots, so the type refuses them.
interface Props
  extends Omit<
    ComponentPropsWithoutRef<"svg">,
    "viewBox" | "fill" | "stroke" | "strokeWidth" | "strokeLinejoin" | "focusable" | "aria-hidden" | "role"
  > {
  className?: string;
}

/**
 * Bibilab mark — a pause button whose negative space reads as an open book.
 * Color is inherited via currentColor; set it on the parent with a text-*
 * class. UI accent colors (pink / blue / sky) must never be applied to it.
 */
export function BrandMark({ className = "h-7 w-7", ...rest }: Props) {
  return (
    <svg
      viewBox="0 0 64 64"
      aria-hidden="true"
      focusable="false"
      className={className}
      {...rest}
    >
      <g fill="currentColor" stroke="currentColor" strokeWidth={6} strokeLinejoin="round">
        <path d="M10 15 Q19 11 27 20 L27 52 L10 52 Z" />
        <path d="M54 15 Q45 11 37 20 L37 52 L54 52 Z" />
      </g>
    </svg>
  );
}
