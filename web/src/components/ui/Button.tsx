import { ComponentPropsWithoutRef, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";

// py-[11px]: button vertical padding — no matching Tailwind spacing step
const base =
  "inline-flex items-center justify-center rounded-2xl transition disabled:cursor-not-allowed disabled:opacity-60";

const variants: Record<Variant, string> = {
  primary:   "border border-transparent bg-linear-to-br from-pink to-blue px-4 py-[11px] text-white hover:brightness-105",
  secondary: "border border-border bg-white/92 px-4 py-[11px] text-ink hover:bg-white",
  ghost:     "border border-blue/18 bg-transparent px-4 py-[11px] text-blue hover:bg-sky/8",
  danger:    "border border-transparent bg-danger px-4 py-[11px] text-white hover:brightness-105",
};

interface Props extends ComponentPropsWithoutRef<"button"> {
  variant?: Variant;
}

export const Button = forwardRef<HTMLButtonElement, Props>(
  ({ variant = "secondary", className = "", ...rest }, ref) => (
    <button
      ref={ref}
      className={`${base} ${variants[variant]} ${className}`.trim()}
      {...rest}
    />
  ),
);
Button.displayName = "Button";
