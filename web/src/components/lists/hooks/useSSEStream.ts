import { useEffect, useRef, useState } from "react";

import type { JobRegistration } from "@/components/jobs/JobActivityProvider";
import type { MessageUI } from "@/components/lists/hooks/useConversationHistory";
import { formatTimestamp, type RagMetadata, type ContentBlock } from "@/lib/chat-utils";
import type { ToolResult } from "@/lib/chat-utils";
import {
  SSE_EVENT_CITATION,
  SSE_EVENT_DELTA,
  SSE_EVENT_DONE,
  SSE_EVENT_ERROR,
  SSE_EVENT_TOOL_RESULT,
} from "@/lib/constants";
import { LANG_STORAGE_KEY } from "@/lib/utils";

type CitationEvent = { type: "citation"; index: number; source_id: string };

interface UseSSEStreamOptions {
  listId: string;
  selectedSourceIds: string[];
  setMessages: React.Dispatch<React.SetStateAction<MessageUI[]>>;
  trackJobs?: (jobs: JobRegistration[]) => void;
  interruptedLabel?: string;
}

interface UseSSEStreamReturn {
  sendMessage: (text: string) => Promise<void>;
  stopStreaming: () => void;
  retryLastMessage: () => void;
  isStreaming: boolean;
}

export function useSSEStream({
  listId,
  selectedSourceIds,
  setMessages,
  trackJobs,
  interruptedLabel = "Interrupted",
}: UseSSEStreamOptions): UseSSEStreamReturn {
  const [isStreaming, setIsStreaming] = useState(false);
  const isStreamingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastUserMessageRef = useRef<string>("");
  const mountedRef = useRef(true);

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

  async function sendMessage(text: string): Promise<void> {
    if (!text) return;
    if (isStreamingRef.current) return;

    lastUserMessageRef.current = text;
    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now()}`;

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
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    safeSetIsStreaming(true);
    isStreamingRef.current = true;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let accBlocks: ContentBlock[] = [];
    let pendingText = "";

    const flushText = () => {
      if (pendingText) {
        accBlocks.push({ type: "text", text: pendingText });
        pendingText = "";
      }
    };

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

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      if (!response.body) throw new Error("Response body is null");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let incomplete = "";

      const processLine = (raw: string) => {
        if (!raw) return;
        let event: { type: string; [key: string]: unknown };
        try {
          event = JSON.parse(raw);
        } catch {
          return;
        }

        if (event.type === SSE_EVENT_DELTA) {
          const content = event.content as string;
          pendingText += content;
          updateAssistantMsg(assistantMsgId, (m) => ({
            content: m.content + content,
            contentBlocks: [...accBlocks, { type: "text", text: pendingText }],
          }));
        } else if (event.type === SSE_EVENT_TOOL_RESULT) {
          const toolName = event.name as string;
          if (!toolName) return;
          if (toolName === "retrieve") {
            const rag = event.result as unknown as RagMetadata;
            updateAssistantMsg(assistantMsgId, { rag });
          } else {
            // generate_report
            const result = event.result as ToolResult;
            if (!result) return;
            const toolCallData = { name: "generate_report", result };
            if (result.job_id && trackJobs) {
              trackJobs([{ id: result.job_id, producer: "artifact", label: result.name, contextKey: listId }]);
            }
            updateAssistantMsg(assistantMsgId, { toolCall: toolCallData });
          }
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
          });
          safeSetIsStreaming(false);
          isStreamingRef.current = false;
        } else if (event.type === SSE_EVENT_ERROR) {
          const errorMsg = event.message as string;
          updateAssistantMsg(assistantMsgId, { isStreaming: false, error: errorMsg });
          safeSetIsStreaming(false);
          isStreamingRef.current = false;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        const chunkText = decoder.decode(value, { stream: !done });
        const hasNewline = chunkText.includes("\n");

        if (hasNewline) {
          const combined = incomplete + chunkText;
          const lines = combined.split("\n");
          incomplete = lines.pop() ?? "";
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              processLine(line.slice(6).trim());
            }
          }
        } else {
          incomplete += chunkText;
        }

        if (done) {
          if (incomplete) {
            if (incomplete.startsWith("data: ")) {
              processLine(incomplete.slice(6).trim());
            }
          }
          break;
        }
      }

      safeSetIsStreaming(false);
      isStreamingRef.current = false;
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        updateAssistantMsg(assistantMsgId, { isStreaming: false, error: interruptedLabel });
      } else {
        updateAssistantMsg(assistantMsgId, { isStreaming: false, error: String(err) });
      }
      safeSetIsStreaming(false);
      isStreamingRef.current = false;
    } finally {
      abortControllerRef.current = null;
    }
  }

  function stopStreaming() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }

  function retryLastMessage() {
    if (lastUserMessageRef.current) {
      const text = lastUserMessageRef.current;
      stopStreaming();
      safeSetIsStreaming(false);
      isStreamingRef.current = false;
      if (mountedRef.current) {
        setMessages((prev) => prev.slice(0, -2));
      }
      void sendMessage(text);
    }
  }

  return { sendMessage, stopStreaming, retryLastMessage, isStreaming };
}
