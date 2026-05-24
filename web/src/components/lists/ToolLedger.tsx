import { METADATA_TOOL_NAME, TOOL_DISPLAY } from "@/lib/tool-display";
import type { RetrievalCall, MetadataCall, PendingRagCall, PendingMetadataCall } from "@/lib/chat-utils";
import { ToolLedgerRow } from "./ToolLedgerRow";

interface ToolLedgerProps {
  ragCalls?: RetrievalCall[];
  metadataCalls?: MetadataCall[];
  pendingRagCalls?: PendingRagCall[];
  pendingMetadataCalls?: PendingMetadataCall[];
  streaming?: boolean;
}

type Step =
  | { key: string; call: RetrievalCall | MetadataCall }
  | { key: string; pending: PendingRagCall | PendingMetadataCall };

export function ToolLedger({
  ragCalls = [],
  metadataCalls = [],
  pendingRagCalls = [],
  pendingMetadataCalls = [],
  streaming = false,
}: ToolLedgerProps) {
  const steps: Step[] = [
    ...ragCalls.map((call, i) => ({ key: `rc${i}`, call })),
    ...metadataCalls.map((call, i) => ({ key: `mc${i}`, call })),
    ...pendingRagCalls.map((p) => ({ key: `pr-${p.id}`, pending: p })),
    ...pendingMetadataCalls.map((p) => ({ key: `pm-${p.id}`, pending: p })),
  ];

  if (steps.length === 0) return null;

  const getConfig = (s: Step) => {
    if ("call" in s) {
      const name = "query_type" in s.call ? METADATA_TOOL_NAME : (s.call as RetrievalCall).tool_name ?? "retrieve";
      return TOOL_DISPLAY[name] ?? TOOL_DISPLAY.retrieve;
    }
    const name = "query_type" in s.pending ? METADATA_TOOL_NAME : (s.pending as PendingRagCall).tool_name ?? "retrieve";
    return TOOL_DISPLAY[name] ?? TOOL_DISPLAY.retrieve;
  };

  const getDotClass = (s: Step) =>
    "call" in s ? "ledger-dot" : "ledger-dot ledger-dot--pending";

  const renderRow = (s: Step) =>
    "call" in s
      ? <ToolLedgerRow config={getConfig(s)} call={s.call} streaming={streaming} />
      : <ToolLedgerRow config={getConfig(s)} pending={s.pending} />;

  if (steps.length === 1) {
    return <div className="w-full flex flex-col">{renderRow(steps[0])}</div>;
  }

  return (
    <div className="w-full ledger-rail flex flex-col gap-1.5">
      {steps.map((s) => (
        <div key={s.key} className="ledger-step">
          <span className={getDotClass(s)} />
          {renderRow(s)}
        </div>
      ))}
    </div>
  );
}
