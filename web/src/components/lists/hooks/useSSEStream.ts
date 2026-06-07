import { useEffect, useRef, useState } from "react";

import type { JobRegistration } from "@/components/jobs/JobActivityProvider";
import type { MessageUI } from "@/components/lists/hooks/useConversationHistory";
import { formatTimestamp, type ContentBlock, type PendingRagCall, type RetrievalCall } from "@/lib/chat-utils";
import { FIND_PASSAGES_TOOL_NAME, READ_SOURCE_TOOL_NAME } from "@/lib/tool-display";
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
    initialAssistantMsgId: string,
  ): Promise<void> {
    // Mutable: on the `meta` event, the server returns its own messageId
    // (a uuid), and we swap the local message's `id` to that uuid so the
    // /debug/messages/{id} fetch matches the dump file on disk. Subsequent
    // updateAssistantMsg calls follow the new id via this reassignment.
    let assistantMsgId = initialAssistantMsgId;
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
        const serverId = event.message_id as string;
        currentAssistantMsgIdRef.current = serverId;
        // Swap the client-generated id for the server-uuid. The dump file
        // is keyed by the server-uuid, so this makes the `</>` button work
        // without a conversation reload. Subsequent updateAssistantMsg
        // calls follow the new id via the closure variable.
        updateAssistantMsg(assistantMsgId, { id: serverId });
        assistantMsgId = serverId;
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
        if (toolName === FIND_PASSAGES_TOOL_NAME) {
          const args = event.arguments as { query: string };
          updateAssistantMsg(assistantMsgId, (m) => ({
            pendingRagCalls: [
              ...m.pendingRagCalls,
              { id, tool_name: FIND_PASSAGES_TOOL_NAME, query: args.query },
            ],
          }));
        } else if (toolName === READ_SOURCE_TOOL_NAME) {
          // read_source has no `query` field; the chip shows the tool label only
          // (read in full). It will resolve to source_id/source_title on tool_result.
          updateAssistantMsg(assistantMsgId, (m) => ({
            pendingRagCalls: [
              ...m.pendingRagCalls,
              { id, tool_name: READ_SOURCE_TOOL_NAME, query: "" },
            ],
          }));
        }
      } else if (event.type === SSE_EVENT_TOOL_RESULT) {
        const toolName = event.name as string;
        if (!toolName) return;
        if (toolName === FIND_PASSAGES_TOOL_NAME) {
          const call = event.result as unknown as RetrievalCall;
          const callId = event.id as string;
          updateAssistantMsg(assistantMsgId, (m) => ({
            rag: { calls: [...(m.rag?.calls ?? []), call] },
            pendingRagCalls: callId
              ? m.pendingRagCalls.filter((p) => p.id !== callId)
              : m.pendingRagCalls,
          }));
          if (!callId) {
            console.warn("find_passages tool_result missing id, pending chip not cleared");
          }
        } else if (toolName === READ_SOURCE_TOOL_NAME) {
          const call = event.result as unknown as RetrievalCall;
          const callId = event.id as string;
          updateAssistantMsg(assistantMsgId, (m) => ({
            rag: { calls: [...(m.rag?.calls ?? []), call] },
            pendingRagCalls: callId
              ? m.pendingRagCalls.filter((p) => p.id !== callId)
              : m.pendingRagCalls,
          }));
          if (!callId) {
            console.warn("read_source tool_result missing id, pending chip not cleared");
          }
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
          // Backend writes the dump at end of `run_chat_turn`, which fires
          // before `done` is yielded to the client. Optimistically set
          // hasDump: true so the </> icon shows immediately. If
          // debug_prompts is off, ChatPanel gates the button off via the
          // `showDebugButton` prop, so the optimistic value is harmless.
          // If the dump write fails (best-effort), the click 404s — an
          // acceptable failure mode for a debug-only feature.
          hasDump: true,
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
        });
        safeSetIsStreaming(false);
        isStreamingRef.current = false;
      } else if (event.type === SSE_EVENT_ERROR) {
        const errorMsg = event.message as string;
        updateAssistantMsg(assistantMsgId, { isStreaming: false, error: errorMsg, pendingRagCalls: [] });
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
      updateAssistantMsg(assistantMsgId, { isStreaming: false, error: String(err), pendingRagCalls: [] });
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
      error: null,
      timestamp: formatTimestamp(new Date().toISOString()),
      pendingRagCalls: [],
      hasDump: false,
    };

    const assistantMsg: MessageUI = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
      contentBlocks: [],
      error: null,
      timestamp: formatTimestamp(new Date().toISOString()),
      rag: null,
      pendingRagCalls: [],
      hasDump: false,
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
      updateAssistantMsg(assistantMsgId, { isStreaming: false, error: String(err), pendingRagCalls: [] });
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
        error: null,
        timestamp: "",
        rag: null,
        pendingRagCalls: [],
        hasDump: false,
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
      updateAssistantMsg(messageId, { isStreaming: false, error: String(err), pendingRagCalls: [] });
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
