import { useState } from "react";
import { DebugHeader } from "./DebugHeader";
import { isMessageEnvelope, isToolCall, isToolDefinition } from "./envelopeHints";
import { JsonTree } from "./JsonTree";

export function DebugDrawer({
  messageId,
  dump,
  onClose,
}: {
  messageId: string;
  dump: unknown;
  onClose: () => void;
}) {
  const [view, setView] = useState<"styled" | "raw">("styled");
  const d = dump as {
    system?: string;
    tools?: unknown[];
    messages?: unknown[];
    response?: { text: string; tool_calls?: unknown[] };
    model?: string;
    timestamp?: string;
  };
  return (
    <div
      className="fixed inset-y-0 right-0 w-1/2 bg-white border-l border-(--color-border) shadow-lg flex flex-col"
      data-testid="debug-drawer"
    >
      <DebugHeader
        messageId={messageId}
        model={d.model}
        timestamp={d.timestamp}
        view={view}
        setView={setView}
        onClose={onClose}
      />
      <div className="flex-1 overflow-y-auto p-6 space-y-3">
        {view === "raw" ? (
          <pre className="text-xs font-mono whitespace-pre-wrap break-words text-(--color-ink)">
            {JSON.stringify(dump, null, 2)}
          </pre>
        ) : (
          <>
            <Section
              title="System Prompt"
              meta={`${d.system?.split("\n").length ?? 0} lines`}
            >
              <JsonTree value={d.system} />
            </Section>
            <Section title="Function catalog" meta={`${d.tools?.length ?? 0} functions`}>
              {(d.tools ?? []).every(isToolDefinition) ? (
                <div className="flex flex-col gap-2">
                  {((d.tools ?? []) as { name: string; description: string; parameters: unknown }[]).map((tool, i) => (
                    <div key={i} className="rounded-lg border border-(--color-border) p-3">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm text-(--color-ink)">{tool.name}</span>
                      </div>
                      {tool.description && (
                        <p className="mt-1 text-xs text-(--color-muted)">{tool.description}</p>
                      )}
                      <details className="mt-2">
                        <summary className="text-2xs text-(--color-muted) cursor-pointer">parameters</summary>
                        <div className="mt-1">
                          <JsonTree value={tool.parameters} />
                        </div>
                      </details>
                    </div>
                  ))}
                </div>
              ) : (
                <JsonTree value={d.tools} />
              )}
            </Section>
            <div className="text-xs uppercase tracking-wider text-(--color-muted) mt-4 mb-2">
              Conversation
            </div>
            {(d.messages ?? []).map((m: unknown, i: number) => (
              <MessageCard key={i} value={m} />
            ))}
            {d.response && <FinalCard response={d.response} />}
          </>
        )}
      </div>
    </div>
  );
}

function Section({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: string;
  children: React.ReactNode;
}) {
  return (
    <details className="rounded-xl border border-(--color-border)" open>
      <summary className="px-4 py-3 flex items-center justify-between cursor-pointer">
        <span className="font-medium text-sm">{title}</span>
        {meta && <span className="text-xs text-(--color-muted)">{meta}</span>}
      </summary>
      <div className="px-4 pb-4">{children}</div>
    </details>
  );
}

function MessageCard({ value }: { value: unknown }) {
  if (!isMessageEnvelope(value)) {
    return (
      <div className="rounded-lg border border-(--color-border) p-3">
        <JsonTree value={value} />
      </div>
    );
  }
  const role = value.role;
  const chipClass =
    role === "user"
      ? "bg-blue-100 text-blue-900"
      : role === "assistant"
        ? "bg-(--color-surface) text-(--color-ink)"
        : "bg-white text-(--color-muted)";
  return (
    <div className="mt-3">
      <div className="flex items-center gap-2 mb-2">
        <span
          className={`text-2xs font-medium uppercase tracking-wide px-2 py-0.5 rounded ${chipClass}`}
        >
          {role}
        </span>
      </div>
      <div className="rounded-xl border border-(--color-border) p-3 bg-white">
        <JsonTree value={value.content} />
        {value.tool_calls && (
          <div className="mt-2 flex flex-wrap gap-2">
            {value.tool_calls.map((tc: unknown, i: number) =>
              isToolCall(tc) ? (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 text-xs bg-(--color-surface) text-(--color-ink) px-2 py-1 rounded"
                >
                  <span className="font-medium">{tc.function.name}</span>
                  <span className="text-(--color-muted) font-mono text-2xs">
                    {tc.id}
                  </span>
                </span>
              ) : null,
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function FinalCard({
  response,
}: {
  response: { text: string; tool_calls?: unknown[] };
}) {
  return (
    <div className="mt-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xs font-medium uppercase tracking-wide px-2 py-0.5 rounded bg-(--color-surface) text-(--color-ink)">
          assistant
        </span>
        <span className="text-xs text-(--color-pink) font-medium">
          FINAL · 用户看到的回答
        </span>
      </div>
      <div className="rounded-xl bg-white p-3 border border-(--color-border) border-l-4 border-l-(--color-pink)">
        <JsonTree value={response.text} />
      </div>
    </div>
  );
}
