import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  ChevronDown,
  MessageSquare,
  MessageSquareOff,
  RotateCcw,
  SendHorizontal,
  Square,
  Trash2,
} from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import type { Source } from "@/lib/types";

const SUGGESTION_CHIPS = [
  "What's the intuition behind backprop?",
  "Compare gradient descent and Adam.",
  "Summarize the diffusion pipeline.",
];

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
  return `${m}m`;
}

function formatSubtitle(sourceCount: number, totalSeconds: number): string {
  const srcLabel = sourceCount === 1 ? "source" : "sources";
  return `${sourceCount} ${srcLabel} · ${formatDuration(totalSeconds)} total`;
}

interface ChatPanelProps {
  selectedSourceIds: string[];
  sources: Source[];
  onSendMessage: (message: string) => void;
}

export function ChatPanel({ selectedSourceIds, sources, onSendMessage }: ChatPanelProps) {
  const { t } = useLanguage();
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasConversation, setHasConversation] = useState(false);
  const [showClearPopover, setShowClearPopover] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messageListRef = useRef<HTMLDivElement>(null);

  const hasSources = selectedSourceIds.length > 0;
  const totalDuration = useMemo(
    () =>
      sources
        .filter((s) => selectedSourceIds.includes(s.id))
        .reduce((acc, s) => acc + (s.duration_seconds ?? 0), 0),
    [sources, selectedSourceIds],
  );

  const placeholder = !hasSources
    ? t("chat.input.placeholder.noSources")
    : !hasConversation
      ? t("chat.input.placeholder.noHistory")
      : isStreaming
        ? t("chat.input.placeholder.streaming")
        : t("chat.input.placeholder.followUp");

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;

    const updateHeight = () => {
      ta.style.height = "auto";
      const scrollHeight = ta.scrollHeight;
      const maxHeight = 88;
      ta.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
      ta.style.overflowY = scrollHeight > maxHeight ? "auto" : "hidden";
    };

    updateHeight();
    ta.addEventListener("input", updateHeight);
    return () => ta.removeEventListener("input", updateHeight);
  }, []);

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

  function scrollToBottom() {
    const list = messageListRef.current;
    if (!list) return;
    list.scrollTop = list.scrollHeight;
  }

  function handleSend() {
    const text = inputValue.trim();
    if (!text || !hasSources || isStreaming) return;
    onSendMessage(text);
    setInputValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.overflowY = "hidden";
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleSuggestionClick(text: string) {
    if (!hasSources || isStreaming) return;
    setInputValue(text);
    onSendMessage(text);
    setInputValue("");
  }

  function handleClearConfirm() {
    setShowClearPopover(false);
    setHasConversation(false);
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="shrink-0 border-b border-border px-4 py-3.5">
        <div className="flex items-center">
          <h2 className="flex-1 font-serif text-lg text-ink">{t("chat.header.title")}</h2>
          <div className="relative">
            <button
              type="button"
              onClick={() => setShowClearPopover((v) => !v)}
              disabled={!hasConversation}
              aria-label="Clear conversation"
              className="flex h-7 w-7 items-center justify-center rounded-lg text-muted transition hover:bg-border disabled:cursor-not-allowed disabled:opacity-30 hover:disabled:bg-transparent"
            >
              <Trash2 size={14} />
            </button>

            {showClearPopover && (
              <div className="absolute right-0 top-9 z-30 w-60 rounded-xl border border-border bg-white p-3.5 shadow-lg">
                <p className="mb-1 text-sm font-semibold text-ink">{t("chat.clearConfirm.title")}</p>
                <p className="mb-3 text-xs text-muted">{t("chat.clearConfirm.body")}</p>
                <div className="flex justify-end gap-1.5">
                  <button
                    type="button"
                    onClick={() => setShowClearPopover(false)}
                    className="rounded-full px-3 py-1.5 text-xs text-muted transition hover:bg-border"
                  >
                    {t("chat.clearConfirm.cancel")}
                  </button>
                  <button
                    type="button"
                    onClick={handleClearConfirm}
                    className="rounded-full bg-ink px-3 py-1.5 text-xs font-medium text-white transition hover:brightness-110"
                  >
                    {t("chat.clearConfirm.clear")}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {hasSources && (
          <div className="mt-0.5 text-xs text-muted font-sans">
            {formatSubtitle(selectedSourceIds.length, totalDuration)}
          </div>
        )}
      </div>

      {/* Message list */}
      <div
        ref={messageListRef}
        className="flex flex-1 flex-col gap-3.5 overflow-y-auto px-4.5 py-4"
        style={{ scrollbarWidth: "thin", scrollbarColor: "var(--color-border) transparent" }}
      >
        {!hasSources ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-surface text-muted">
              <MessageSquareOff size={26} />
            </div>
            <h3 className="m-0 font-serif text-lg text-ink">{t("chat.empty.noSources.title")}</h3>
            <p className="m-0 max-w-[260px] text-sm text-muted">
              {t("chat.empty.noSources.hint")}
            </p>
          </div>
        ) : !hasConversation ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-pink/20 to-sky/18 text-blue">
              <MessageSquare size={26} />
            </div>
            <h3 className="m-0 font-serif text-lg text-ink">{t("chat.empty.noHistory.title")}</h3>
            <p className="m-0 max-w-[260px] text-sm text-muted">
              {t("chat.empty.noHistory.hint")}
            </p>

            <div className="mt-2 grid w-full max-w-[280px] gap-1.5">
              {SUGGESTION_CHIPS.map((text) => (
                <button
                  key={text}
                  type="button"
                  onClick={() => handleSuggestionClick(text)}
                  className="text-left text-xs text-ink bg-white/64 border border-border rounded-xl px-3 py-2 transition hover:bg-white hover:border-blue/25"
                >
                  <span className="mr-1.5 text-blue font-mono text-xs">→</span>
                  {text}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {/* Scroll-to-bottom button */}
        {showScrollButton && (
          <button
            type="button"
            onClick={scrollToBottom}
            aria-label="Scroll to bottom"
            className="absolute bottom-14 left-1/2 flex h-8 w-8 -translate-x-1/2 items-center justify-center rounded-full border border-border bg-white shadow-lg transition hover:-translate-x-1/2 hover:-translate-y-px"
          >
            <ChevronDown size={16} className="text-ink" />
          </button>
        )}
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-border bg-white/55 px-3.5 py-3">
        <div className="relative">
          <div className="relative border border-border bg-white/90 rounded-2xl transition focus-within:border-blue/25 focus-within:ring-2 focus-within:ring-sky/18">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={!hasSources || isStreaming}
              rows={1}
              className="w-full resize-none border-0 bg-transparent px-3.5 py-2.5 pr-11 text-sm text-ink placeholder:text-muted focus:outline-none disabled:cursor-not-allowed disabled:bg-surface/70 disabled:text-muted"
              style={{ maxHeight: "88px" }}
            />

            <button
              type="button"
              onClick={isStreaming ? () => setIsStreaming(false) : handleSend}
              disabled={!hasSources || (!isStreaming && !inputValue.trim())}
              aria-label={isStreaming ? "Stop" : "Send"}
              className={`absolute bottom-1 right-1 flex h-7 w-7 items-center justify-center rounded-full text-white transition ${
                isStreaming
                  ? "bg-pink hover:brightness-110"
                  : "bg-blue hover:brightness-105 disabled:bg-border disabled:cursor-not-allowed"
              }`}
            >
              {isStreaming ? <Square size={12} fill="currentColor" /> : <SendHorizontal size={14} />}
            </button>
          </div>
        </div>

        <div className="mt-1.5 flex justify-between px-0.5 font-mono text-xs text-muted">
          <span>{t("chat.input.hint.enter")}</span>
          <span>{t("chat.input.hint.shiftEnter")}</span>
        </div>
      </div>
    </div>
  );
}
