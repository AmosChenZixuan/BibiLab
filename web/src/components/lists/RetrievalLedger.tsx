import type { RetrievalCall, PendingRagCall, PendingMetadataCall } from "@/lib/chat-utils";
import { RetrievalLedgerRow, type RowVariant } from "./RetrievalLedgerRow";

interface RetrievalLedgerProps {
  calls: RetrievalCall[];
  pendingRetrieve?: PendingRagCall[];
  pendingMetadata?: PendingMetadataCall[];
}

function callVariant(call: RetrievalCall): RowVariant {
  if (call.reused_from_prior_call_id != null) return "reused";
  if ((call.context ?? []).length === 0 && call.dropped_by_gate > 0) return "empty";
  return "default";
}

const DOT_CLASS: Record<RowVariant, string> = {
  default: "ledger-dot",
  empty: "ledger-dot",
  reused: "ledger-dot ledger-dot--faint",
  pending: "ledger-dot ledger-dot--pending",
};

type Step =
  | { key: string; variant: RowVariant; call: RetrievalCall }
  | { key: string; variant: "pending"; pending: PendingRagCall | PendingMetadataCall };

export function RetrievalLedger({ calls = [], pendingRetrieve = [], pendingMetadata = [] }: RetrievalLedgerProps) {
  const steps: Step[] = [
    ...calls.map((call, i) => ({ key: `c${i}`, variant: callVariant(call), call })),
    ...pendingRetrieve.map((p) => ({ key: `pr-${p.id}`, variant: "pending" as const, pending: p })),
    ...pendingMetadata.map((p) => ({ key: `pm-${p.id}`, variant: "pending" as const, pending: p })),
  ];

  if (steps.length === 0) return null;

  const renderRow = (s: Step) =>
    "call" in s
      ? <RetrievalLedgerRow variant={s.variant} call={s.call} />
      : <RetrievalLedgerRow variant="pending" pending={s.pending} />;

  // Single step: no rail (a lone dot reads as noise). Multi step: connected rail.
  if (steps.length === 1) {
    return <div className="w-full flex flex-col">{renderRow(steps[0])}</div>;
  }

  return (
    <div className="w-full ledger-rail flex flex-col gap-1.5">
      {steps.map((s) => (
        <div key={s.key} className="ledger-step">
          <span className={DOT_CLASS[s.variant]} />
          {renderRow(s)}
        </div>
      ))}
    </div>
  );
}
