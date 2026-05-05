export const SSE_EVENT_DELTA = "delta" as const;
export const SSE_EVENT_DONE = "done" as const;
export const SSE_EVENT_ERROR = "error" as const;
export const SSE_EVENT_TOOL_RESULT = "tool_result" as const;
export type SSEEventType =
  | typeof SSE_EVENT_DELTA
  | typeof SSE_EVENT_DONE
  | typeof SSE_EVENT_ERROR
  | typeof SSE_EVENT_TOOL_RESULT;
