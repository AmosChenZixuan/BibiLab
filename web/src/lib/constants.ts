export const SSE_EVENT_DELTA = "delta" as const;
export const SSE_EVENT_DONE = "done" as const;
export const SSE_EVENT_ERROR = "error" as const;
export const SSE_EVENT_TOOL_RESULT = "tool_result" as const;
export const SSE_EVENT_TOOL_CALL_START = "tool_call_start" as const;
export const SSE_EVENT_CITATION = "citation" as const;
export const SSE_EVENT_CANCELLED = "cancelled" as const;
export const SSE_EVENT_META = "meta" as const;
export const SSE_EVENT_RAG = "rag" as const;
export type SSEEventType =
  | typeof SSE_EVENT_DELTA
  | typeof SSE_EVENT_DONE
  | typeof SSE_EVENT_ERROR
  | typeof SSE_EVENT_TOOL_RESULT
  | typeof SSE_EVENT_TOOL_CALL_START
  | typeof SSE_EVENT_CITATION
  | typeof SSE_EVENT_CANCELLED
  | typeof SSE_EVENT_RAG;
