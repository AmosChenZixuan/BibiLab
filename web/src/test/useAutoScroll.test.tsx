import { act, render } from "@testing-library/react";
import { useLayoutEffect } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { useAutoScroll } from "@/components/lists/hooks/useAutoScroll";
import type { MessageUI } from "@/components/lists/hooks/useConversationHistory";

function makeMessage(id: string): MessageUI {
  return {
    id,
    role: "assistant",
    content: "hi",
    isStreaming: false,
    contentBlocks: [],
    error: null,
    timestamp: "2026-01-01T00:00:00Z",
    rag: null,
    pendingRagCalls: [],
  };
}

interface Metrics { scrollHeight: number; clientHeight: number; scrollTop: number }

const DEFAULT_METRICS: Metrics = { scrollHeight: 1000, clientHeight: 500, scrollTop: 500 };

function Harness({
  isLoadingHistory,
  messages,
  metrics,
  scrollTo,
}: {
  isLoadingHistory: boolean;
  messages: MessageUI[];
  metrics: Metrics;
  scrollTo: { (opts?: ScrollToOptions): void; (x: number, y: number): void };
}) {
  const result = useAutoScroll({ isLoadingHistory, messages });
  // Attach the ref'd div and override its layout-readonly properties so the
  // hook's scroll listener and scrollToBottom() see the metrics we choose.
  useLayoutEffect(() => {
    const el = result.messageListRef.current;
    if (!el) return;
    Object.defineProperty(el, "scrollHeight", { configurable: true, get: () => metrics.scrollHeight });
    Object.defineProperty(el, "clientHeight", { configurable: true, get: () => metrics.clientHeight });
    Object.defineProperty(el, "scrollTop", { configurable: true, get: () => metrics.scrollTop, set: () => {} });
    el.scrollTo = scrollTo;
  });
  return <div ref={result.messageListRef} data-testid="list" />;
}

function setup(initial: {
  isLoadingHistory?: boolean;
  messages?: MessageUI[];
  metrics?: Metrics;
}) {
  const isLoadingHistory = initial.isLoadingHistory ?? false;
  const messages = initial.messages ?? [];
  const metrics = initial.metrics ?? DEFAULT_METRICS;
  const scrollToSpy = vi.fn();

  const utils = render(
    <Harness
      isLoadingHistory={isLoadingHistory}
      messages={messages}
      metrics={metrics}
      scrollTo={scrollToSpy}
    />,
  );
  const list = utils.container.querySelector("[data-testid='list']") as HTMLDivElement;

  function rerender(next: { isLoadingHistory?: boolean; messages?: MessageUI[]; metrics?: Metrics }) {
    utils.rerender(
      <Harness
        isLoadingHistory={next.isLoadingHistory ?? isLoadingHistory}
        messages={next.messages ?? messages}
        metrics={next.metrics ?? metrics}
        scrollTo={scrollToSpy}
      />,
    );
  }

  return { list, scrollToSpy, rerender };
}

function fireScroll(list: HTMLDivElement) {
  act(() => {
    list.dispatchEvent(new Event("scroll"));
  });
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useAutoScroll", () => {
  test("mount with messages scrolls instantly (no smooth animation)", () => {
    const { scrollToSpy } = setup({ messages: [makeMessage("m1")] });
    act(() => {
      vi.runAllTimers();
    });
    expect(scrollToSpy).toHaveBeenCalledTimes(1);
    expect(scrollToSpy).toHaveBeenCalledWith({ top: 1000, behavior: "auto" });
  });

  test("new message while at bottom scrolls to follow", () => {
    const { scrollToSpy, rerender } = setup({ messages: [makeMessage("m1")] });
    act(() => {
      vi.runAllTimers();
    });
    scrollToSpy.mockClear();

    rerender({ messages: [makeMessage("m1"), makeMessage("m2")] });
    act(() => {
      vi.runAllTimers();
    });
    expect(scrollToSpy).toHaveBeenCalledTimes(1);
    expect(scrollToSpy).toHaveBeenCalledWith({ top: 1000, behavior: "auto" });
  });

  test("new message while scrolled up does NOT scroll", () => {
    const { scrollToSpy, list, rerender } = setup({
      messages: [makeMessage("m1")],
      metrics: { scrollHeight: 1000, clientHeight: 500, scrollTop: 0 },
    });
    act(() => {
      vi.runAllTimers();
    });
    // initial mount scrolled; clear so the assertion below is only about the next change
    scrollToSpy.mockClear();
    // simulate the user scrolling up (the hook's listener will flip isAtBottom to false)
    fireScroll(list);

    rerender({ messages: [makeMessage("m1"), makeMessage("m2")] });
    act(() => {
      vi.runAllTimers();
    });
    expect(scrollToSpy).not.toHaveBeenCalled();
  });

  test("scrolling back to bottom re-enables auto-scroll on next message", () => {
    const { scrollToSpy, list, rerender } = setup({
      messages: [makeMessage("m1")],
      metrics: { scrollHeight: 1000, clientHeight: 500, scrollTop: 0 },
    });
    act(() => {
      vi.runAllTimers();
    });
    scrollToSpy.mockClear();
    fireScroll(list); // scrolled up
    rerender({ messages: [makeMessage("m1"), makeMessage("m2")] });
    act(() => {
      vi.runAllTimers();
    });
    expect(scrollToSpy).not.toHaveBeenCalled();

    // user scrolls back to bottom: top 1000, height 500, view 500 → at bottom
    rerender({ metrics: { scrollHeight: 1000, clientHeight: 500, scrollTop: 500 } });
    fireScroll(list);
    rerender({ messages: [makeMessage("m1"), makeMessage("m2"), makeMessage("m3")] });
    act(() => {
      vi.runAllTimers();
    });
    expect(scrollToSpy).toHaveBeenCalledTimes(1);
  });

  test("empty messages never scrolls", () => {
    const { scrollToSpy } = setup({ messages: [] });
    act(() => {
      vi.runAllTimers();
    });
    expect(scrollToSpy).not.toHaveBeenCalled();
  });

  test("isLoadingHistory=true suppresses scroll even with messages", () => {
    const { scrollToSpy } = setup({
      messages: [makeMessage("m1")],
      isLoadingHistory: true,
    });
    act(() => {
      vi.runAllTimers();
    });
    expect(scrollToSpy).not.toHaveBeenCalled();
  });
});
