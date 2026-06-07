import { useState } from "react";
import { useLanguage } from "@/app/LanguageContext";
import { TEST_IDS } from "@/lib/test-ids";
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
    response?: { text: string };
    model?: string;
    timestamp?: string;
  };
  // Split messages into "prior conversation" (everything before the last
  // user message) and "this turn" (the last user message + everything the
  // LLM emitted in response). This makes the tool chain + final answer
  // prominent by default while keeping prior context one click away.
  const messages = (d.messages ?? []) as Array<{ role: string }>;
  let lastUserIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      lastUserIdx = i;
      break;
    }
  }
  const priorConversation = lastUserIdx === -1 ? messages : messages.slice(0, lastUserIdx);
  const thisTurn = lastUserIdx === -1 ? [] : messages.slice(lastUserIdx);
  return (
    <div
      className="fixed inset-y-0 right-0 w-1/2 bg-white border-l border-(--color-border) shadow-lg flex flex-col z-(--z-modal)"
      data-testid={TEST_IDS.debugDrawer}
    >
      <DebugHeader
        messageId={messageId}
        model={d.model}
        timestamp={d.timestamp}
        view={view}
        setView={setView}
        onClose={onClose}
        dump={dump}
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
            {priorConversation.length > 0 && (
              <details className="rounded-xl border border-(--color-border)">
                <summary className="px-4 py-3 flex items-center justify-between cursor-pointer">
                  <span className="font-medium text-sm">Prior conversation</span>
                  <span className="text-xs text-(--color-muted)">
                    {priorConversation.length} message{priorConversation.length === 1 ? "" : "s"}
                  </span>
                </summary>
                <div className="px-4 pb-4 space-y-3">
                  {priorConversation.map((m, i) => (
                    <MessageCard key={`prior-${i}`} value={m} />
                  ))}
                </div>
              </details>
            )}
            {thisTurn.length > 0 && (
              <>
                <div className="text-xs uppercase tracking-wider text-(--color-muted) mt-4 mb-2">
                  本轮调用链
                </div>
                {thisTurn.map((m, i) => (
                  <MessageCard key={`turn-${i}`} value={m} />
                ))}
              </>
            )}
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
    <details className="rounded-xl border border-(--color-border)">
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
  // Skip content rendering for null/empty — assistant messages with only
  // tool_calls have content: null (the synthesis text lives in response.text
  // / FinalCard). Showing "—" would be noise.
  const hasContent = value.content !== null && value.content !== undefined && value.content !== "";
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
        {role === "tool" && typeof value.content === "string" ? (
          <ToolMessageContent content={value.content} />
        ) : (
          hasContent && <JsonTree value={value.content} />
        )}
        {value.tool_calls && value.tool_calls.length > 0 && (
          <div className="mt-2 flex flex-col gap-2">
            {value.tool_calls.map((tc: unknown, i: number) =>
              isToolCall(tc) ? <ToolCallBlock key={i} tc={tc} /> : null,
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ToolMessageContent({ content }: { content: string }) {
  // Tool result content is plain text with newlines (not markdown —
  // we don't want stray `*` / `_` / `#` to be interpreted as formatting).
  // Preserves whitespace; collapses if too long.
  const PREVIEW_LEN = 280;
  const [expanded, setExpanded] = useState(false);
  const long = content.length > PREVIEW_LEN;
  const preview = long && !expanded ? content.slice(0, PREVIEW_LEN) + "…" : content;
  return (
    <div>
      <div className="whitespace-pre-wrap text-xs leading-relaxed text-(--color-ink)">
        {preview}
      </div>
      {long && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1 text-2xs text-(--color-muted) hover:text-(--color-ink) cursor-pointer"
        >
          {expanded ? "▾ collapse" : `▸ expand (${content.length} chars)`}
        </button>
      )}
    </div>
  );
}

function ToolCallBlock({
  tc,
}: {
  tc: { id: string; type: string; function: { name: string; arguments: unknown } };
}) {
  // function.arguments can be a JSON string (OpenAI wire format) or a
  // structured object/list (Anthropic tool_use). Normalize to a parseable
  // value for pretty-printing in the code block.
  const argsJson: string | null = (() => {
    let value: unknown = tc.function.arguments;
    if (typeof value === "string") {
      try {
        value = JSON.parse(value);
      } catch {
        return value as string;  // unparseable string — show as-is
      }
    }
    if (value === null || value === undefined) return null;
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return null;
    }
  })();
  return (
    <>
      <span className="inline-flex items-center gap-1.5 text-xs bg-(--color-surface) text-(--color-ink) px-2 py-1 rounded">
        <span className="font-medium">{tc.function.name}</span>
        <span className="font-mono text-2xs text-(--color-muted)">{tc.id}</span>
      </span>
      {argsJson !== null && (
        <JsonCodeBlock json={argsJson} />
      )}
    </>
  );
}

function JsonCodeBlock({ json }: { json: string }) {
  // Lightweight tokenizer: match keys (string + colon), string values,
  // numbers/booleans/null, and the rest as plain punctuation. Match the
  // mock's key/str color split (blue / emerald).
  const tokens: Array<{ text: string; cls: string }> = [];
  const re = /("(?:[^"\\]|\\.)*"\s*:)|("(?:[^"\\]|\\.)*")|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([{}\[\],:])/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(json))) {
    if (m[1]) tokens.push({ text: m[1], cls: "text-(--color-blue) font-medium" });
    else if (m[2]) tokens.push({ text: m[2], cls: "text-emerald-700" });
    else if (m[3]) tokens.push({ text: m[3], cls: "text-(--color-muted) italic" });
    else if (m[4]) tokens.push({ text: m[4], cls: "text-(--color-pink)" });
    else if (m[5]) tokens.push({ text: m[5], cls: "text-(--color-muted)" });
  }
  return (
    <pre className="mt-2 overflow-x-auto rounded-md bg-(--color-surface) px-3 py-2.5 font-mono text-xs leading-relaxed text-(--color-ink)">
      {tokens.map((t, i) => (
        <span key={i} className={t.cls}>
          {t.text}
        </span>
      ))}
    </pre>
  );
}

function FinalCard({ response }: { response: { text: string } }) {
  const { t } = useLanguage();
  return (
    <div className="mt-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xs font-medium uppercase tracking-wide px-2 py-0.5 rounded bg-(--color-surface) text-(--color-ink)">
          assistant
        </span>
        <span className="text-xs text-(--color-pink) font-medium">
          FINAL · {t("debug.final")}
        </span>
      </div>
      <div className="rounded-xl bg-white p-3 border border-(--color-border) border-l-4 border-l-(--color-pink)">
        <JsonTree value={response.text} />
      </div>
    </div>
  );
}
