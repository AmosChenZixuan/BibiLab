import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import {
  formatTimestamp,
  parseCitations,
  type Citation,
  type ToolCallData,
  type ToolResult,
} from "@/lib/chat-utils";

export interface MessageUI {
  id: string;
  role: "user" | "assistant";
  content: string;
  isStreaming: boolean;
  citations: Citation[];
  toolCall: ToolCallData | null;
  error: string | null;
  timestamp: string;
}

export function useConversationHistory(listId: string | undefined, hasSources: boolean) {
  const [messages, setMessages] = useState<MessageUI[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  useEffect(() => {
    if (!listId || !hasSources) return;
    let cancelled = false;
    setIsLoadingHistory(true);
    setMessages([]);

    api
      .getConversation(listId)
      .then((data) => {
        if (cancelled || !data) return;
        if (data.messages.length === 0) return;
        const loaded: MessageUI[] = data.messages.map((m) => {
          const { citations, cleanContent } = parseCitations(m.content);
          let toolCall: ToolCallData | null = null;
          const tcList = m.metadata?.tool_calls as Array<{ name: string; result?: ToolResult }> | undefined;
          const tc = tcList?.[0];
          if (tc?.result) {
            toolCall = { name: tc.name, result: tc.result };
          }
          return {
            id: m.id,
            role: m.role as "user" | "assistant",
            content: cleanContent,
            isStreaming: false,
            citations,
            toolCall,
            error: null,
            timestamp: formatTimestamp(m.created_at),
          };
        });
        setMessages(loaded);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setIsLoadingHistory(false);
      });

    return () => {
      cancelled = true;
    };
  }, [listId, hasSources]);

  return { messages, isLoadingHistory };
}
