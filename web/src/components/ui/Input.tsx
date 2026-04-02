import { ComponentPropsWithoutRef } from "react";

type InputSize = "md" | "sm";

const sizes: Record<InputSize, string> = {
  md: "px-3.5 py-3",
  sm: "px-3 py-2.5 h-11 min-h-11",
};

interface Props extends ComponentPropsWithoutRef<"input"> {
  inputSize?: InputSize;
}

export function Input({ inputSize = "md", className = "", ...rest }: Props) {
  return (
    <input
      className={`w-full rounded-2xl border border-border bg-white/92 text-ink outline-none transition focus:border-blue/45 focus:ring-2 focus:ring-sky/18 ${sizes[inputSize]} ${className}`.trim()}
      {...rest}
    />
  );
}
