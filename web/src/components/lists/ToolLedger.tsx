import { FIND_PASSAGES_TOOL_NAME, TOOL_DISPLAY } from "@/lib/tool-display";
import type { RetrievalCall, PendingRagCall } from "@/lib/chat-utils";
import { ToolLedgerRow } from "./ToolLedgerRow";

interface ToolLedgerProps {
  ragCalls?: RetrievalCall[];
  pendingRagCalls?: PendingRagCall[];
  streaming?: boolean;
}

type Step =
  | { key: string; call: RetrievalCall }
  | { key: string; pending: PendingRagCall };

export function ToolLedger({
  ragCalls = [],
  pendingRagCalls = [],
  streaming = false,
}: ToolLedgerProps) {
  const steps: Step[] = [
    ...ragCalls.map((call, i) => ({ key: `rc${i}`, call })),
    ...pendingRagCalls.map((p) => ({ key: `pr-${p.id}`, pending: p })),
  ];

  if (steps.length === 0) return null;

  const getConfig = (s: Step) => {
    if ("call" in s) {
      return TOOL_DISPLAY[s.call.tool_name] ?? TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME];
    }
    return TOOL_DISPLAY[s.pending.tool_name] ?? TOOL_DISPLAY[FIND_PASSAGES_TOOL_NAME];
  };

  const renderRow = (s: Step) =>
    "call" in s
      ? <ToolLedgerRow config={getConfig(s)} call={s.call} streaming={streaming} />
      : <ToolLedgerRow config={getConfig(s)} pending={s.pending} />;

  if (steps.length === 1) {
    return <div className="w-full flex flex-col">{renderRow(steps[0])}</div>;
  }

  return (
    <div className="relative w-full flex flex-col gap-1.5">
      <span aria-hidden="true" className="rail-left absolute top-2.5 bottom-2.5 w-px bg-border" />
      {steps.map((s) => {
        const isPending = !("call" in s);
        return (
          <div key={s.key} className="relative pl-4.5">
            <span
              className={`absolute left-0 top-1.5 size-1.5 rounded-full ${
                isPending ? "bg-blue" : "bg-muted"
              }`}
            />
            {renderRow(s)}
          </div>
        );
      })}
    </div>
  );
}
