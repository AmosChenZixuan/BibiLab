import type { RetrievalCall, PendingRagCall, PendingMetadataCall } from "@/lib/chat-utils";
import { RetrievalLedgerRow } from "./RetrievalLedgerRow";

interface RetrievalLedgerProps {
  calls: RetrievalCall[];
  pendingRetrieve?: PendingRagCall[];
  pendingMetadata?: PendingMetadataCall[];
}

export function RetrievalLedger({ calls = [], pendingRetrieve = [], pendingMetadata = [] }: RetrievalLedgerProps) {
  if (calls.length === 0 && pendingRetrieve.length === 0 && pendingMetadata.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-1">
      {calls.map((call, i) => {
        const variant = call.reused_from_prior_call_id != null ? "reused" : call.context.length === 0 ? "empty" : "default";
        return <RetrievalLedgerRow key={i} variant={variant} call={call} />;
      })}
      {pendingRetrieve.map((p) => (
        <RetrievalLedgerRow key={`pending-${p.id}`} variant="pending" pending={p} />
      ))}
      {pendingMetadata.map((p) => (
        <RetrievalLedgerRow key={`pending-meta-${p.id}`} variant="pending" pending={p} />
      ))}
    </div>
  );
}
