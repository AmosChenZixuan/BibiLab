import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import {
  formatTimestamp,
  stripLegacyTokens,
  type ContentBlock,
  type PendingRagCall,
  type RagMetadata,
  type ToolCallData,
  type ToolResult,
} from "@/lib/chat-utils";

export interface MessageUI {
  id: string;
  role: "user" | "assistant";
  content: string;
  isStreaming: boolean;
  contentBlocks: ContentBlock[];
  toolCall: ToolCallData | null;
  error: string | null;
  timestamp: string;
  rag: RagMetadata | null;
  pendingRagCalls: PendingRagCall[];
}

export function useConversationHistory(
  listId: string | undefined,
  hasSources: boolean,
  interruptedLabel: string,
  stoppedLabel: string,
) {
  const [messages, setMessages] = useState<MessageUI[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activeStreamMessageId, setActiveStreamMessageId] = useState<string | null>(null);

  useEffect(() => {
    if (!listId || !hasSources) return;
    let cancelled = false;
    setIsLoadingHistory(true);
    setMessages([]);
    setLoadError(null);

    api
      .getConversation(listId)
      .then((data) => {
        if (cancelled || !data) return;
        setActiveStreamMessageId(data.conversation?.active_stream_message_id ?? null);
        if (!data.messages?.length) return;
        const loaded: MessageUI[] = data.messages.map((m) => {
          let contentBlocks: ContentBlock[];
          let displayContent = "";

          if (m.metadata?.content_blocks) {
            contentBlocks = m.metadata.content_blocks as ContentBlock[];
          } else if (m.content) {
            const stripped = stripLegacyTokens(m.content);
            contentBlocks = [{ type: "text", text: stripped }];
            displayContent = stripped;
          } else {
            contentBlocks = [];
          }

          let toolCall: ToolCallData | null = null;
          const tcList = m.metadata?.tool_calls as Array<{ name: string; result?: ToolResult }> | undefined;
          const tc = tcList?.[0];
          if (tc?.result) {
            toolCall = { name: tc.name, result: tc.result };
          }
          let rag: RagMetadata | null = null;
          const rawRag = m.metadata?.rag as Record<string, unknown> | undefined;
          if (rawRag?.calls) {
            rag = rawRag as unknown as RagMetadata;
          }
          return {
            id: m.id,
            role: m.role as "user" | "assistant",
            content: displayContent,
            isStreaming: false,
            contentBlocks,
            toolCall,
            error:
              m.error ??
              (m.status === "failed" ? interruptedLabel : m.status === "cancelled" ? stoppedLabel : null),
            timestamp: formatTimestamp(m.created_at),
            rag,
            pendingRagCalls: [],
          };
        });
        setMessages(loaded);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(String(err));
      })
      .finally(() => {
        if (!cancelled) setIsLoadingHistory(false);
      });

    return () => {
      cancelled = true;
    };
  }, [listId, hasSources]);

  return { messages, isLoadingHistory, loadError, activeStreamMessageId };
}
