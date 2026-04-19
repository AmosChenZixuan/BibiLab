import { ComponentPropsWithoutRef } from "react";

type Variant = "app" | "workspace";

const variants: Record<Variant, string> = {
  app:       "rounded-3xl border border-border bg-white/80 p-5 shadow-lg",
  workspace: "overflow-hidden rounded-3xl border border-border bg-white/76 shadow-lg",
};

interface Props extends ComponentPropsWithoutRef<"div"> {
  variant?: Variant;
}

export function Panel({ variant = "app", className = "", ...rest }: Props) {
  return (
    <div className={`${variants[variant]} ${className}`.trim()} {...rest} />
  );
}
