import { ComponentPropsWithoutRef, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "md" | "sm";

// py-[11px]: button vertical padding — no matching Tailwind spacing step
const base =
  "inline-flex items-center justify-center rounded-2xl transition disabled:cursor-not-allowed disabled:opacity-60";

const variants: Record<Variant, string> = {
  primary:   "border border-transparent bg-linear-to-br from-pink to-blue text-white hover:brightness-105",
  secondary: "border border-border bg-white/92 text-ink hover:bg-white",
  ghost:     "border border-blue/18 bg-transparent text-blue hover:bg-sky/8",
  danger:    "border border-transparent bg-danger text-white hover:brightness-105",
};

const sizes: Record<Size, string> = {
  md: "px-4 py-[11px]",
  sm: "px-3 py-1.5 text-sm",
};

interface Props extends ComponentPropsWithoutRef<"button"> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, Props>(
  ({ variant = "secondary", size = "md", className = "", ...rest }, ref) => (
    <button
      ref={ref}
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`.trim()}
      {...rest}
    />
  ),
);
Button.displayName = "Button";
