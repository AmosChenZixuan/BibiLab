import { Loader2 } from "lucide-react";
import { useLanguage } from "@/app/LanguageContext";
import type { RetrievalCall, PendingRagCall, PendingMetadataCall } from "@/lib/chat-utils";
import { RetrievalLedgerRow, type RowVariant } from "./RetrievalLedgerRow";

interface RetrievalLedgerProps {
  calls: RetrievalCall[];
  pendingRetrieve?: PendingRagCall[];
  pendingMetadata?: PendingMetadataCall[];
  /** Rewriter stage is in flight (LLM call before any tool dispatch).
   * Renders a leading "analyzing query" row so the user sees activity. */
  rewriterPending?: boolean;
  /** Message still streaming — rows render collapsed, non-expandable until the
   * final `rag` event delivers the complete context[]. */
  streaming?: boolean;
}

function callVariant(call: RetrievalCall): RowVariant {
  // context is absent (not []) on the streaming payload — only an explicit
  // empty array (persisted, everything gated out) is the "empty" variant.
  if (call.context != null && call.context.length === 0 && call.dropped_by_gate > 0) return "empty";
  return "default";
}

const DOT_CLASS: Record<RowVariant, string> = {
  default: "ledger-dot",
  empty: "ledger-dot",
  pending: "ledger-dot ledger-dot--pending",
};

type Step =
  | { key: string; variant: RowVariant; call: RetrievalCall }
  | { key: string; variant: "pending"; pending: PendingRagCall | PendingMetadataCall };

function RewriterRow() {
  const { t } = useLanguage();
  return (
    <div className="flex items-center gap-1.5 py-0.5 text-xs text-muted opacity-70">
      <Loader2 size={12} className="animate-spin shrink-0" />
      <span>{t("chat.ledger.rewriterPending")}</span>
    </div>
  );
}

export function RetrievalLedger({
  calls = [],
  pendingRetrieve = [],
  pendingMetadata = [],
  rewriterPending = false,
  streaming = false,
}: RetrievalLedgerProps) {
  const steps: Step[] = [
    ...calls.map((call, i) => ({ key: `c${i}`, variant: callVariant(call), call })),
    ...pendingRetrieve.map((p) => ({ key: `pr-${p.id}`, variant: "pending" as const, pending: p })),
    ...pendingMetadata.map((p) => ({ key: `pm-${p.id}`, variant: "pending" as const, pending: p })),
  ];

  if (steps.length === 0 && !rewriterPending) return null;

  const renderRow = (s: Step) =>
    "call" in s
      ? <RetrievalLedgerRow variant={s.variant} call={s.call} streaming={streaming} />
      : <RetrievalLedgerRow variant="pending" pending={s.pending} />;

  const totalRows = steps.length + (rewriterPending ? 1 : 0);

  // Single step: no rail (a lone dot reads as noise). Multi step: connected rail.
  if (totalRows === 1) {
    if (rewriterPending) return <div className="w-full flex flex-col"><RewriterRow /></div>;
    return <div className="w-full flex flex-col">{renderRow(steps[0])}</div>;
  }

  return (
    <div className="w-full ledger-rail flex flex-col gap-1.5">
      {rewriterPending && (
        <div className="ledger-step">
          <span className="ledger-dot ledger-dot--pending" />
          <RewriterRow />
        </div>
      )}
      {steps.map((s) => (
        <div key={s.key} className="ledger-step">
          <span className={DOT_CLASS[s.variant]} />
          {renderRow(s)}
        </div>
      ))}
    </div>
  );
}
