import { ComponentPropsWithoutRef } from "react";

type Variant = "app" | "workspace";

const variants: Record<Variant, string> = {
  app:       "rounded-[--radius-card] border border-border bg-white/80 p-5 shadow-lg",
  workspace: "overflow-hidden rounded-[--radius-feature] border border-border bg-white/76 shadow-lg",
};

interface Props extends ComponentPropsWithoutRef<"div"> {
  variant?: Variant;
}

export function Panel({ variant = "app", className = "", ...rest }: Props) {
  return (
    <div className={`${variants[variant]} ${className}`.trim()} {...rest} />
  );
}

export function PanelTitle({ className = "", ...rest }: ComponentPropsWithoutRef<"h2">) {
  return (
    <h2
      className={`m-0 border-b border-border px-5 py-4.5 font-serif text-2xl ${className}`.trim()}
      {...rest}
    />
  );
}

export function PanelBody({ className = "", ...rest }: ComponentPropsWithoutRef<"div">) {
  return (
    <div className={`grid gap-4 px-5 py-4.5 ${className}`.trim()} {...rest} />
  );
}
