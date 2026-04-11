import { ComponentPropsWithoutRef } from "react";
import { ChevronDown } from "lucide-react";

interface Props extends ComponentPropsWithoutRef<"select"> {}

export function Select({ className = "", ...rest }: Props) {
  return (
    <div className="relative w-full">
      <select
        className={`h-11 w-full appearance-none rounded-lg border border-divider bg-white px-3 pr-10 text-charcoal outline-none transition focus:border-pink focus:ring-3 focus:ring-pink/30 disabled:cursor-not-allowed disabled:opacity-60 ${className}`.trim()}
        {...rest}
      />
      <span className="pointer-events-none absolute inset-y-0 right-3 inline-flex items-center text-icon-secondary">
        <ChevronDown className="size-4" aria-hidden="true" />
      </span>
    </div>
  );
}
