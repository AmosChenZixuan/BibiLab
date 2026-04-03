import { ComponentPropsWithoutRef, ReactNode } from "react";

const alignments = {
  start:  "items-start",
  center: "items-center",
};

interface Props extends Omit<ComponentPropsWithoutRef<"div">, "children"> {
  label:    string;
  hint?:    string;
  htmlFor?: string;
  align?:   keyof typeof alignments;
  children: ReactNode;
}

export function SettingsField({
  label,
  hint,
  htmlFor,
  align = "start",
  children,
  className = "",
  ...rest
}: Props) {
  return (
    <div
      className={`flex flex-wrap ${alignments[align]} gap-x-5 gap-y-2 bg-white/36 px-4 py-3 ${className}`.trim()}
      {...rest}
    >
      <div className="grid min-w-48 flex-1 basis-60 gap-1">
        <label className="text-sm font-semibold" htmlFor={htmlFor}>{label}</label>
        {hint ? <p className="text-sm leading-5 text-muted">{hint}</p> : null}
      </div>
      <div className="w-full min-w-56 flex-none md:w-80">
        {children}
      </div>
    </div>
  );
}
