import { useEffect, useRef, useState } from "react";

import type { JobRegistration } from "@/components/jobs/JobActivityProvider";
import type { MessageUI } from "@/components/lists/hooks/useConversationHistory";
import { formatTimestamp, type ContentBlock, type PendingMetadataCall, type PendingRagCall, type RetrievalCall, type Mode } from "@/lib/chat-utils";
import type { ToolResult } from "@/lib/chat-utils";
import {
  SSE_EVENT_CANCELLED,
  SSE_EVENT_CITATION,
  SSE_EVENT_DELTA,
  SSE_EVENT_DONE,
  SSE_EVENT_ERROR,
  SSE_EVENT_META,
  SSE_EVENT_RAG,
  SSE_EVENT_TOOL_CALL_START,
  SSE_EVENT_TOOL_RESULT,
} from "@/lib/constants";
import { LANG_STORAGE_KEY } from "@/lib/utils";

type CitationEvent = { type: "citation"; index: number; source_id: string; chunk_ids: string[] };

interface UseSSEStreamOptions {
  listId: string;
  selectedSourceIds: string[];
  messages: MessageUI[];
  setMessages: React.Dispatch<React.SetStateAction<MessageUI[]>>;
  trackJobs?: (jobs: JobRegistration[]) => void;
  interruptedLabel?: string;
  stoppedLabel?: string;
}

interface UseSSEStreamReturn {
  sendMessage: (text: string) => Promise<void>;
  stopStreaming: () => void;
  retryMessage: (assistantMessageId: string) => void;
  reattach: (messageId: string) => Promise<void>;
  isStreaming: boolean;
}

export function useSSEStream({
  listId,
  selectedSourceIds,
  messages,
  setMessages,
  trackJobs,
  interruptedLabel = "Interrupted",
  stoppedLabel = "Stopped",
}: UseSSEStreamOptions): UseSSEStreamReturn {
  const [isStreaming, setIsStreaming] = useState(false);
  const isStreamingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentAssistantMsgIdRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortControllerRef.current?.abort();
    };
  }, []);

  function updateAssistantMsg(
    id: string,
    patchOrFn: Partial<MessageUI> | ((current: MessageUI) => Partial<MessageUI>),
  ) {
    if (!mountedRef.current) return;
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id !== id) return m;
        const patch = typeof patchOrFn === "function" ? patchOrFn(m) : patchOrFn;
        return { ...m, ...patch };
      }),
    );
  }

  function safeSetIsStreaming(val: boolean) {
    if (!mountedRef.current) return;
    setIsStreaming(val);
  }

  async function consumeSSE(
    response: Response,
    assistantMsgId: string,
  ): Promise<void> {
    if (!response.body) throw new Error("Response body is null");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let incomplete = "";

    let accBlocks: ContentBlock[] = [];
    let pendingText = "";

    const flushText = () => {
      if (!pendingText) return;
      const parts = pendingText.split(/\n{2,}/);
      for (let j = 0; j < parts.length; j++) {
        if (parts[j]) {
          accBlocks.push({ type: "text", text: parts[j] });
        }
        if (j < parts.length - 1) {
          accBlocks.push({ type: "paragraph_break" });
        }
      }
      pendingText = "";
    };

    function processSSEEvent(event: { type: string; [key: string]: unknown }): void {
      if (event.type === SSE_EVENT_META) {
        currentAssistantMsgIdRef.current = event.message_id as string;
        return;
      }
      if (event.type === SSE_EVENT_DELTA) {
        const content = event.content as string;
        pendingText += content;
        updateAssistantMsg(assistantMsgId, (m) => ({
          content: m.content + content,
          contentBlocks: [...accBlocks, { type: "text", text: pendingText }],
        }));
      } else if (event.type === SSE_EVENT_TOOL_CALL_START) {
        const toolName = event.name as string;
        const id = event.id as string;
        if (!id || !toolName || !event.arguments) {
          console.warn("tool_call_start missing required fields", event);
          return;
        }
        if (toolName === "retrieve" || toolName === "survey" || toolName === "retrieve_scoped") {
          const args = event.arguments as { query: string };
          const mode: Mode = toolName === "survey" ? "survey" : "narrow";
          updateAssistantMsg(assistantMsgId, (m) => ({
            pendingRagCalls: [
              ...m.pendingRagCalls,
              { id, query: args.query, mode, tool_name: toolName },
            ],
          }));
        } else if (toolName === "query_list_metadata") {
          const args = event.arguments as { query_type: string };
          updateAssistantMsg(assistantMsgId, (m) => ({
            pendingMetadataCalls: [
              ...m.pendingMetadataCalls,
              { id, query_type: args.query_type },
            ],
          }));
        }
      } else if (event.type === SSE_EVENT_TOOL_RESULT) {
        const toolName = event.name as string;
        if (!toolName) return;
        if (toolName === "retrieve") {
          const call = event.result as unknown as RetrievalCall;
          const callId = event.id as string;
          updateAssistantMsg(assistantMsgId, (m) => ({
            rag: { calls: [...(m.rag?.calls ?? []), call] },
            pendingRagCalls: callId
              ? m.pendingRagCalls.filter((p) => p.id !== callId)
              : m.pendingRagCalls,
          }));
          if (!callId) {
            console.warn("retrieve tool_result missing id, pending chip not cleared");
          }
        } else if (toolName === "query_list_metadata") {
          const callId = event.id as string;
          const args = event.result as Record<string, unknown> | undefined;
          updateAssistantMsg(assistantMsgId, (m) => ({
            pendingMetadataCalls: callId
              ? m.pendingMetadataCalls.filter((p) => p.id !== callId)
              : m.pendingMetadataCalls,
            metadataCalls: [
              ...(m.metadataCalls ?? []),
              {
                name: "query_list_metadata",
                query_type: (args?.query_type as string) ?? "unknown",
                result: event.result ?? {},
              },
            ],
          }));
          if (!callId) {
            console.warn("query_list_metadata tool_result missing id, pending chip not cleared");
          }
        } else if (toolName === "generate_report") {
          const result = event.result as ToolResult;
          if (!result) return;
          const toolCallData = { name: "generate_report", result };
          if (result.job_id && trackJobs) {
            trackJobs([{ id: result.job_id, producer: "artifact", label: result.name, contextKey: listId }]);
          }
          updateAssistantMsg(assistantMsgId, { toolCall: toolCallData });
        }
      } else if (event.type === SSE_EVENT_RAG) {
        // Final authoritative ledger (persisted shape, with context[]).
        // Replaces the incremental tool_result entries so expand works
        // post-stream without a refresh.
        const calls = Array.isArray(event.calls) ? (event.calls as RetrievalCall[]) : [];
        if (!Array.isArray(event.calls)) {
          console.warn("SSE: rag event missing or invalid calls field", event);
        }
        updateAssistantMsg(assistantMsgId, { rag: { calls } });
      } else if (event.type === SSE_EVENT_CITATION) {
        flushText();
        const citation = event as unknown as CitationEvent;
        accBlocks.push(citation);
        updateAssistantMsg(assistantMsgId, { contentBlocks: [...accBlocks] });
      } else if (event.type === SSE_EVENT_DONE) {
        flushText();
        updateAssistantMsg(assistantMsgId, {
          isStreaming: false,
          contentBlocks: [...accBlocks],
          pendingRagCalls: [],
          pendingMetadataCalls: [],
        });
        safeSetIsStreaming(false);
        isStreamingRef.current = false;
      } else if (event.type === SSE_EVENT_CANCELLED) {
        flushText();
        updateAssistantMsg(assistantMsgId, {
          isStreaming: false,
          error: stoppedLabel,
          contentBlocks: [...accBlocks],
          pendingRagCalls: [],
          pendingMetadataCalls: [],
        });
        safeSetIsStreaming(false);
        isStreamingRef.current = false;
      } else if (event.type === SSE_EVENT_ERROR) {
        const errorMsg = event.message as string;
        updateAssistantMsg(assistantMsgId, { isStreaming: false, error: errorMsg, pendingRagCalls: [], pendingMetadataCalls: [] });
        safeSetIsStreaming(false);
        isStreamingRef.current = false;
      }
    }

    try {
      while (true) {
        const { done, value } = await reader.read();
        const chunkText = decoder.decode(value, { stream: !done });
        const hasNewline = chunkText.includes("\n");

        if (hasNewline) {
          const combined = incomplete + chunkText;
          const lines = combined.split("\n");
          incomplete = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (!raw) continue;
            try {
              processSSEEvent(JSON.parse(raw));
            } catch (e) {
              console.warn("SSE: failed to parse event", raw, e);
            }
            // Yield to React between events so streaming animation is visible on reattach
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        } else {
          incomplete += chunkText;
        }

        if (done) {
          if (incomplete.startsWith("data: ")) {
            const raw = incomplete.slice(6).trim();
            if (raw) {
              try {
                processSSEEvent(JSON.parse(raw));
              } catch (e) {
                console.warn("SSE: failed to parse final event", raw, e);
              }
            }
          }
          break;
        }
      }

      updateAssistantMsg(assistantMsgId, { isStreaming: false });
      safeSetIsStreaming(false);
      isStreamingRef.current = false;
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        // AbortError from cancel — producer's 'cancelled' event handles UI state
        return;
      }
      updateAssistantMsg(assistantMsgId, { isStreaming: false, error: String(err), pendingRagCalls: [], pendingMetadataCalls: [] });
      safeSetIsStreaming(false);
      isStreamingRef.current = false;
    }
  }

  async function sendMessage(text: string): Promise<void> {
    if (!text) return;
    if (isStreamingRef.current) return;

    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now()}`;
    currentAssistantMsgIdRef.current = assistantMsgId;

    const userMsg: MessageUI = {
      id: userMsgId,
      role: "user",
      content: text,
      isStreaming: false,
      contentBlocks: [],
      rag: null,
      toolCall: null,
      error: null,
      timestamp: formatTimestamp(new Date().toISOString()),
      pendingRagCalls: [],
      pendingMetadataCalls: [],
      metadataCalls: null,
    };

    const assistantMsg: MessageUI = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
      contentBlocks: [],
      toolCall: null,
      error: null,
      timestamp: formatTimestamp(new Date().toISOString()),
      rag: null,
      pendingRagCalls: [],
      pendingMetadataCalls: [],
      metadataCalls: null,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    safeSetIsStreaming(true);
    isStreamingRef.current = true;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`/api/lists/${listId}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-UI-Lang": localStorage.getItem(LANG_STORAGE_KEY) ?? "en",
        },
        body: JSON.stringify({
          message: text,
          source_ids: selectedSourceIds,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          if (body.detail) detail = body.detail;
        } catch { /* use status-only fallback */ }
        throw new Error(detail);
      }

      await consumeSSE(response, assistantMsgId);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        // Server-side cancel handles UI state
        return;
      }
      updateAssistantMsg(assistantMsgId, { isStreaming: false, error: String(err), pendingRagCalls: [], pendingMetadataCalls: [] });
      safeSetIsStreaming(false);
      isStreamingRef.current = false;
    } finally {
      abortControllerRef.current = null;
    }
  }

  async function reattach(messageId: string): Promise<void> {
    if (isStreamingRef.current) return;

    currentAssistantMsgIdRef.current = messageId;

    setMessages((prev) => {
      const existing = prev.find((m) => m.id === messageId);
      if (existing) {
        // If message has substantive content from history, it's complete — skip reattach
        const hasRealContent = existing.contentBlocks?.some(
          (b) => b.type !== "text" || b.text.length > 0,
        );
        if (existing.content || hasRealContent) {
          return prev;
        }
        return prev.map((m) => (m.id === messageId ? { ...m, isStreaming: true, error: null } : m));
      }
      const newMsg: MessageUI = {
        id: messageId,
        role: "assistant",
        content: "",
        isStreaming: true,
        contentBlocks: [],
        toolCall: null,
        error: null,
        timestamp: "",
        rag: null,
        pendingRagCalls: [],
        pendingMetadataCalls: [],
        metadataCalls: null,
      };
      return [...prev, newMsg];
    });
    safeSetIsStreaming(true);
    isStreamingRef.current = true;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(
        `/api/lists/${listId}/chat/${messageId}/stream`,
        { signal: controller.signal },
      );

      if (response.status === 204) {
        updateAssistantMsg(messageId, { isStreaming: false });
        safeSetIsStreaming(false);
        isStreamingRef.current = false;
        return;
      }

      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          if (body.detail) detail = body.detail;
        } catch { /* use status-only fallback */ }
        throw new Error(detail);
      }

      await consumeSSE(response, messageId);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      updateAssistantMsg(messageId, { isStreaming: false, error: String(err), pendingRagCalls: [], pendingMetadataCalls: [] });
      safeSetIsStreaming(false);
      isStreamingRef.current = false;
    } finally {
      abortControllerRef.current = null;
    }
  }

  async function stopStreaming() {
    const msgId = currentAssistantMsgIdRef.current;
    if (!msgId) {
      abortControllerRef.current?.abort();
      return;
    }
    try {
      const res = await fetch(`/api/lists/${listId}/chat/${msgId}/cancel`, { method: "POST" });
      if (!res.ok) {
        // Server doesn't know this message (e.g. not yet persisted, or already
        // evicted) — fall back to client-side abort so the UI doesn't hang.
        abortControllerRef.current?.abort();
      }
    } catch (e) {
      console.warn("Cancel API call failed, falling back to client-side abort", e);
      abortControllerRef.current?.abort();
    }
  }

  async function retryMessage(assistantMessageId: string) {
    const msgs = messagesRef.current;
    const asstIndex = msgs.findIndex((m) => m.id === assistantMessageId);
    if (asstIndex === -1) return;

    let userIndex = -1;
    for (let i = asstIndex - 1; i >= 0; i--) {
      if (msgs[i].role === "user") {
        userIndex = i;
        break;
      }
    }
    if (userIndex === -1) return;

    const text = msgs[userIndex].content;
    if (!text) return;

    await stopStreaming();
    safeSetIsStreaming(false);
    isStreamingRef.current = false;
    void sendMessage(text);
  }

  return { sendMessage, stopStreaming, retryMessage, reattach, isStreaming };
}
