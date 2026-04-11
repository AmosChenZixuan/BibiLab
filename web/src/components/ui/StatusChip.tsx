import { ComponentPropsWithoutRef } from "react";

type Status = "ok" | "error" | "unavailable" | "neutral";

const statusColors: Record<Status, string> = {
  ok:          "bg-success text-white",
  error:       "bg-error text-white",
  unavailable: "bg-soft-gray text-secondary-text",
  neutral:     "bg-sky-blue text-white",
};

interface Props extends ComponentPropsWithoutRef<"span"> {
  status?: Status;
}

export function StatusChip({ status = "neutral", className = "", ...rest }: Props) {
  return (
    <span
      className={`inline-flex items-center rounded-full border border-border px-2.5 py-1.5 text-sm capitalize ${statusColors[status]} ${className}`.trim()}
      {...rest}
    />
  );
}
