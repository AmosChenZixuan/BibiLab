export function isMessageEnvelope(
  v: unknown,
): v is {
  role: string;
  content?: unknown;
  tool_calls?: unknown[];
  tool_call_id?: string;
} {
  return typeof v === "object" && v !== null && "role" in v;
}

export function isToolCall(v: unknown): v is {
  id: string;
  type: string;
  function: { name: string; arguments: unknown };
} {
  return (
    typeof v === "object" &&
    v !== null &&
    "id" in v &&
    "type" in v &&
    "function" in v
  );
}

export function isToolDefinition(
  v: unknown,
): v is { name: string; description: string; parameters: unknown } {
  return (
    typeof v === "object" &&
    v !== null &&
    typeof (v as { name?: unknown }).name === "string" &&
    "parameters" in (v as object)
  );
}
