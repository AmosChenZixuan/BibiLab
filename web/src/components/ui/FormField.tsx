import { ComponentPropsWithoutRef, ReactNode } from "react";

interface Props extends Omit<ComponentPropsWithoutRef<"label">, "children"> {
  label: string;
  hint?: string;
  children: ReactNode;
}

// Renders as <label> — browser implicitly associates with first interactive child.
// Use for single-input fields (input, select, textarea).
export function FormField({ label, hint, children, className = "", ...rest }: Props) {
  return (
    <label className={`grid gap-1.5 ${className}`.trim()} {...rest}>
      <span className="text-sm font-semibold text-charcoal">{label}</span>
      {children}
      {hint ? <span className="text-sm leading-5 text-secondary-text">{hint}</span> : null}
    </label>
  );
}
