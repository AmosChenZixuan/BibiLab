import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import {
  formatTimestamp,
  stripLegacyTokens,
  type ContentBlock,
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
}

export function useConversationHistory(listId: string | undefined, hasSources: boolean) {
  const [messages, setMessages] = useState<MessageUI[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

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
        if (!data.messages?.length) return;
        const loaded: MessageUI[] = data.messages.map((m) => {
          let contentBlocks: ContentBlock[];
          let displayContent = "";

          if (m.metadata?.content_blocks) {
            contentBlocks = m.metadata.content_blocks as ContentBlock[];
          } else {
            const stripped = stripLegacyTokens(m.content);
            contentBlocks = [{ type: "text", text: stripped }];
            displayContent = stripped;
          }

          let toolCall: ToolCallData | null = null;
          const tcList = m.metadata?.tool_calls as Array<{ name: string; result?: ToolResult }> | undefined;
          const tc = tcList?.[0];
          if (tc?.result) {
            toolCall = { name: tc.name, result: tc.result };
          }
          return {
            id: m.id,
            role: m.role as "user" | "assistant",
            content: displayContent ?? "",
            isStreaming: false,
            contentBlocks,
            toolCall,
            error: null,
            timestamp: formatTimestamp(m.created_at),
            rag: (m.metadata?.rag as RagMetadata | null) ?? null,
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

  return { messages, isLoadingHistory, loadError };
}
