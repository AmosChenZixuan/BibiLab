import { ComponentPropsWithoutRef, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "md" | "sm";

// py-[11px]: button vertical padding — no matching Tailwind spacing step
const base =
  "inline-flex items-center justify-center rounded-full text-[13px] font-medium tracking-[-0.01em] transition disabled:cursor-not-allowed disabled:opacity-60";

const variants: Record<Variant, string> = {
  primary:   "border border-transparent bg-linear-to-br from-pink to-blue text-white shadow-[0_10px_24px_rgba(91,127,170,0.2)] hover:-translate-y-px hover:brightness-105",
  secondary: "border border-border bg-white/92 text-ink hover:bg-white",
  ghost:     "border border-blue/18 bg-white/55 text-blue hover:bg-sky/8",
  danger:    "border border-transparent bg-danger text-white shadow-[0_10px_24px_rgba(141,29,44,0.18)] hover:-translate-y-px hover:brightness-105",
};

const sizes: Record<Size, string> = {
  md: "px-3.5 py-2.5",
  sm: "px-3 py-1.5 text-sm",
};

interface Props extends ComponentPropsWithoutRef<"button"> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, Props>(
  ({ variant = "secondary", size = "md", className = "", type = "button", ...rest }, ref) => (
    <button
      ref={ref}
      type={type}
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`.trim()}
      {...rest}
    />
  ),
);
Button.displayName = "Button";
