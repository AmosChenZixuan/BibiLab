import { ComponentPropsWithoutRef, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "md" | "sm";

const base =
  "inline-flex items-center justify-center rounded-full text-button transition " +
  "disabled:cursor-not-allowed " +
  "focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-pink focus-visible:ring-offset-2";

const variants: Record<Variant, string> = {
  primary:
    "bg-pink text-white border-transparent " +
    "hover:bg-pink-hover active:bg-pink-pressed " +
    "disabled:bg-divider-gray disabled:text-cta-disabled-text",
  secondary:
    "bg-sky-blue text-white border-transparent " +
    "hover:bg-sky-blue-hover active:bg-sky-blue-pressed " +
    "disabled:bg-divider-gray disabled:text-cta-disabled-text",
  ghost:
    "bg-sky-blue/20 border-secondary-text/18 text-link-blue " +
    "hover:bg-sky-blue/40",
  danger:
    "bg-error text-white border-transparent " +
    "hover:brightness-95 " +
    "disabled:bg-divider-gray disabled:text-cta-disabled-text",
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
