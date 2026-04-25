import { useEffect, useRef, useState } from "react";

import type { MessageUI } from "@/components/lists/hooks/useConversationHistory";

interface UseAutoScrollOptions {
  isLoadingHistory: boolean;
  isStreaming: boolean;
  messages: MessageUI[];
}

interface UseAutoScrollReturn {
  showScrollButton: boolean;
  messageListRef: React.RefObject<HTMLDivElement>;
  scrollToBottom: () => void;
}

export function useAutoScroll({
  isLoadingHistory,
  isStreaming,
  messages,
}: UseAutoScrollOptions): UseAutoScrollReturn {
  const [showScrollButton, setShowScrollButton] = useState(false);
  const messageListRef = useRef<HTMLDivElement>(null);

  function scrollToBottom() {
    const list = messageListRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }

  useEffect(() => {
    if (!isLoadingHistory && messages.length > 0) {
      scrollToBottom();
    }
  }, [isLoadingHistory, messages]);

  useEffect(() => {
    const list = messageListRef.current;
    if (!list) return;

    function onScroll(this: HTMLDivElement) {
      const { scrollTop, scrollHeight, clientHeight } = this;
      setShowScrollButton(scrollHeight - scrollTop - clientHeight > 80);
    }

    list.addEventListener("scroll", onScroll, { passive: true });
    return () => list.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!isStreaming) return;
    const id = setTimeout(() => scrollToBottom(), 0);
    return () => clearTimeout(id);
  }, [isStreaming, messages]);

  return { showScrollButton, messageListRef, scrollToBottom };
}
