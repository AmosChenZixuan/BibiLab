import { ComponentPropsWithoutRef } from "react";
import { ChevronDown } from "lucide-react";

interface Props extends ComponentPropsWithoutRef<"select"> {}

export function Select({ className = "", ...rest }: Props) {
  return (
    <div className="relative w-full">
      <select
        className={`h-11 w-full appearance-none rounded-xl border border-border bg-white/92 px-3 pr-10 text-ink outline-none transition focus:border-blue/45 focus:ring-2 focus:ring-sky/18 disabled:cursor-not-allowed disabled:opacity-60 ${className}`.trim()}
        {...rest}
      />
      <span className="pointer-events-none absolute inset-y-0 right-3 inline-flex items-center text-muted">
        <ChevronDown className="size-4" aria-hidden="true" />
      </span>
    </div>
  );
}
