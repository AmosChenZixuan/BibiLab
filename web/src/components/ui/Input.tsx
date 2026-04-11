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
      className={`w-full rounded-lg border border-border bg-white text-charcoal outline-none transition placeholder:text-secondary-text focus:border-meta-blue focus:ring-3 focus:ring-meta-blue/30 ${sizes[inputSize]} ${className}`.trim()}
      {...rest}
    />
  );
}
