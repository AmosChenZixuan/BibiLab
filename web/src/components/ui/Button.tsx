import { ComponentPropsWithoutRef, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "md" | "sm";

const base =
  "inline-flex items-center justify-center rounded-full text-button transition " +
  "disabled:cursor-not-allowed " +
  "focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-meta-blue focus-visible:ring-offset-2";

const variants: Record<Variant, string> = {
  primary:
    "bg-meta-blue text-white border-transparent " +
    "hover:bg-meta-blue-hover active:bg-meta-blue-pressed " +
    "disabled:bg-[#DEE3E9] disabled:text-[#8595A4]",
  secondary:
    "bg-white/92 border-2 border-[rgba(10,19,23,0.12)] text-charcoal/50 " +
    "hover:border-[rgba(10,19,23,0.24)] hover:bg-[--color-divider-gray] hover:text-white",
  ghost:
    "bg-white/55 border border-[--color-secondary-text]/18 text-meta-blue " +
    "hover:bg-sky-blue-light",
  danger:
    "bg-error text-white border-transparent " +
    "hover:brightness-95 " +
    "disabled:bg-[#DEE3E9] disabled:text-[#8595A4]",
};

const sizes: Record<Size, string> = {
  md: "px-3.5 py-2.5",
  sm: "px-3 py-1.5",
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
