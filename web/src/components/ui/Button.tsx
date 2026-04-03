import { ComponentPropsWithoutRef, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "md" | "sm";

const base =
  "inline-flex items-center justify-center rounded-full text-sm font-medium tracking-tight transition disabled:cursor-not-allowed disabled:opacity-60";

const variants: Record<Variant, string> = {
  primary:   "border border-transparent bg-linear-to-br from-pink to-blue text-white shadow-lg hover:-translate-y-px hover:brightness-105",
  secondary: "border border-border bg-white/92 text-ink hover:bg-white",
  ghost:     "border border-blue/18 bg-white/55 text-blue hover:bg-sky/8",
  danger:    "border border-transparent bg-rose-900 text-white shadow-lg hover:-translate-y-px hover:brightness-105",
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
