export const CHAT_MODE_AUTO = "auto" as const;
export const CHAT_MODE_FOCUSED = "focused" as const;
export const CHAT_MODE_BROAD = "broad" as const;
export type ChatMode = typeof CHAT_MODE_AUTO | typeof CHAT_MODE_FOCUSED | typeof CHAT_MODE_BROAD;

export const SSE_EVENT_DELTA = "delta" as const;
export const SSE_EVENT_DONE = "done" as const;
export const SSE_EVENT_ERROR = "error" as const;
export const SSE_EVENT_TOOL_RESULT = "tool_result" as const;
export const SSE_EVENT_CLEAR_TEXT = "clear_text" as const;
export const SSE_EVENT_RAG_META = "rag_meta" as const;
export type SSEEventType =
  | typeof SSE_EVENT_DELTA
  | typeof SSE_EVENT_DONE
  | typeof SSE_EVENT_ERROR
  | typeof SSE_EVENT_TOOL_RESULT
  | typeof SSE_EVENT_CLEAR_TEXT
  | typeof SSE_EVENT_RAG_META;
