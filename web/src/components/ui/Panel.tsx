import { ComponentPropsWithoutRef } from "react";

type Variant = "app" | "workspace";

const variants: Record<Variant, string> = {
  app:       "rounded-card border border-border bg-surface p-5 shadow-card",
  workspace: "overflow-hidden rounded-3xl border border-border bg-white/76 shadow-card",
};

interface Props extends ComponentPropsWithoutRef<"div"> {
  variant?: Variant;
}

export function Panel({ variant = "app", className = "", ...rest }: Props) {
  return (
    <div className={`${variants[variant]} ${className}`.trim()} {...rest} />
  );
}
