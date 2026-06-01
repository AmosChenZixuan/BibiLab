import { useEffect, useRef, useState } from "react";

import type { MessageUI } from "@/components/lists/hooks/useConversationHistory";

interface UseAutoScrollOptions {
  isLoadingHistory: boolean;
  messages: MessageUI[];
}

interface UseAutoScrollReturn {
  isAtBottom: boolean;
  messageListRef: React.RefObject<HTMLDivElement>;
  scrollToBottom: () => void;
}

const AT_BOTTOM_THRESHOLD_PX = 80;

export function useAutoScroll({
  isLoadingHistory,
  messages,
}: UseAutoScrollOptions): UseAutoScrollReturn {
  // Initial `true`: the user hasn't scrolled on first load, so by definition they
  // are at the bottom. This lets a single effect cover both first-load and
  // subsequent content additions without branching.
  const [isAtBottom, setIsAtBottom] = useState(true);
  const messageListRef = useRef<HTMLDivElement>(null);

  function scrollToBottom() {
    const list = messageListRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: "auto" });
  }

  // Single source of truth for isAtBottom.
  useEffect(() => {
    const list = messageListRef.current;
    if (!list) return;

    function onScroll(this: HTMLDivElement) {
      const { scrollTop, scrollHeight, clientHeight } = this;
      setIsAtBottom(scrollHeight - scrollTop - clientHeight <= AT_BOTTOM_THRESHOLD_PX);
    }

    list.addEventListener("scroll", onScroll, { passive: true });
    return () => list.removeEventListener("scroll", onScroll);
  }, []);

  // Single auto-scroll effect, gated on "user is at bottom". Covers initial
  // load, streaming tokens, new user messages, compression, and reattach —
  // any path that mutates the `messages` array.
  useEffect(() => {
    if (isLoadingHistory) return;
    if (messages.length === 0) return;
    if (!isAtBottom) return;
    const id = setTimeout(() => scrollToBottom(), 0);
    return () => clearTimeout(id);
  }, [isLoadingHistory, messages, isAtBottom]);

  return { isAtBottom, messageListRef, scrollToBottom };
}
