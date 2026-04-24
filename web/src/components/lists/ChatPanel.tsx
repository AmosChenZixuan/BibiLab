import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  AlertCircle,
  ChevronDown,
  FileText,
  MessageSquare,
  MessageSquareOff,
  RotateCcw,
  SendHorizontal,
  Square,
  Trash2,
} from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import type { Source } from "@/lib/types";
import { api } from "@/lib/api";
import {
  autoResize,
  formatSubtitle,
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

interface ChatPanelProps {
  selectedSourceIds: string[];
  sources: Source[];
  listId: string;
  onArtifactGenerated: (artifactId: string, type: string, sourceIds: string[]) => void;
}

export function ChatPanel({
  selectedSourceIds,
  sources,
  listId,
  onArtifactGenerated,
}: ChatPanelProps) {
  const { t } = useLanguage();
  const { trackJobs } = useJobActivity();
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<MessageUI[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [showClearPopover, setShowClearPopover] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messageListRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastUserMessageRef = useRef<string>("");

  const hasSources = selectedSourceIds.length > 0;
  const hasConversation = messages.length > 0;
  const selectedSourceIdsSet = useMemo(() => new Set(selectedSourceIds), [selectedSourceIds]);
  const totalDuration = useMemo(
    () =>
      sources
        .filter((s) => selectedSourceIdsSet.has(s.id))
        .reduce((acc, s) => acc + (s.duration_seconds ?? 0), 0),
    [sources, selectedSourceIdsSet],
  );

  const placeholder = !hasSources
    ? t("chat.input.placeholder.noSources")
    : !hasConversation
      ? t("chat.input.placeholder.noHistory")
      : isStreaming
        ? t("chat.input.placeholder.streaming")
        : t("chat.input.placeholder.followUp");

  function scrollToBottom() {
    const list = messageListRef.current;
    if (!list) return;
    list.scrollTop = list.scrollHeight;
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

  useEffect(() => {
    if (!isStreaming) return;
    const id = setTimeout(() => scrollToBottom(), 0);
    return () => clearTimeout(id);
  }, [isStreaming, messages]);

  async function handleSend(overrideText?: string) {
    const text = (overrideText ?? inputValue).trim();
    if (!text || !hasSources || isStreaming) return;
    setInputValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.overflowY = "hidden";
    }

    lastUserMessageRef.current = text;
    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now()}`;

    const userMsg: MessageUI = {
      id: userMsgId,
      role: "user",
      content: text,
      isStreaming: false,
      citations: [],
      toolCall: null,
      error: null,
      timestamp: formatTimestamp(new Date().toISOString()),
    };

    const assistantMsg: MessageUI = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
      citations: [],
      toolCall: null,
      error: null,
      timestamp: formatTimestamp(new Date().toISOString()),
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);
    scrollToBottom();

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`/api/lists/${listId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-UI-Lang": localStorage.getItem("bibilab-lang") ?? "en" },
        body: JSON.stringify({ message: text, source_ids: selectedSourceIds }),
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

        if (event.type === "clear_text") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, content: "" } : m,
            ),
          );
        } else if (event.type === "delta") {
          const content = event.content as string;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + content }
                : m,
            ),
          );
        } else if (event.type === "tool_result") {
          const result = event.result as ToolResult;
          const toolCallData = { name: "generate_report", result };
          if (result.job_id) {
            trackJobs([{ id: result.job_id, producer: "artifact", label: result.type, contextKey: listId }]);
          }
          onArtifactGenerated(result.artifact_id, result.type, selectedSourceIds);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, toolCall: toolCallData }
                : m,
            ),
          );
        } else if (event.type === "done") {
          setMessages((prev) =>
            prev.map((m) => {
              if (m.id !== assistantMsgId) return m;
              const { citations, cleanContent } = parseCitations(m.content);
              return { ...m, isStreaming: false, content: cleanContent, citations };
            }),
          );
          setIsStreaming(false);
        } else if (event.type === "error") {
          const errorMsg = event.message as string;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, isStreaming: false, error: errorMsg }
                : m,
            ),
          );
          setIsStreaming(false);
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

      setIsStreaming(false);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, isStreaming: false, error: t("chat.interrupted") }
              : m,
          ),
        );
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, isStreaming: false, error: String(err) }
              : m,
          ),
        );
      }
      setIsStreaming(false);
    } finally {
      abortControllerRef.current = null;
    }
  }

  function handleStop() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }

  function handleRetry() {
    if (lastUserMessageRef.current) {
      setMessages((prev) => prev.slice(0, -2));
      void handleSend(lastUserMessageRef.current);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      isStreaming ? handleStop() : handleSend();
    }
  }

  async function handleClearConfirm() {
    setShowClearPopover(false);
    try {
      await api.deleteConversation(listId);
    } catch {
      // non-fatal
    }
    setMessages([]);
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
              aria-label={t("chat.clearConfirm.title")}
              className="flex h-7 w-7 items-center justify-center rounded-lg text-muted transition hover:bg-border disabled:cursor-not-allowed disabled:opacity-30 hover:disabled:bg-transparent"
            >
              <Trash2 size={14} />
            </button>

            {showClearPopover && (
              <div className="popover absolute right-0 top-9 z-30 w-60 rounded-xl border border-border bg-white p-3.5 shadow-lg">
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
            {formatSubtitle(t, selectedSourceIds.length, totalDuration)}
          </div>
        )}
      </div>

      {/* Message list */}
      <div
        ref={messageListRef}
        role="region"
        aria-label={t("chat.header.title")}
        className={`flex flex-1 flex-col gap-3.5 overflow-y-auto px-4.5 py-4 ${showClearPopover ? "opacity-50" : ""}`}
        style={{ scrollbarWidth: "thin", scrollbarColor: "var(--color-border) transparent" }}
      >
        {!hasSources ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-surface text-muted">
              <MessageSquareOff size={26} />
            </div>
            <h3 className="m-0 font-serif text-lg text-ink">{t("chat.empty.noSources.title")}</h3>
            <p className="m-0 max-w-xs text-sm text-muted">
              {t("chat.empty.noSources.hint")}
            </p>
          </div>
        ) : !hasConversation && !isLoadingHistory ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-pink/20 to-sky/18 text-blue">
              <MessageSquare size={26} />
            </div>
            <h3 className="m-0 font-serif text-lg text-ink">{t("chat.empty.noHistory.title")}</h3>
            <p className="m-0 max-w-xs text-sm text-muted">
              {t("chat.empty.noHistory.hint")}
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3.5">
            {messages.map((msg) => (
              <div key={msg.id} className={`msg ${msg.role}`}>
                {msg.role === "user" ? (
                  <>
                    <div className="bubble bubble-user">{msg.content}</div>
                    <span className="ts">{msg.timestamp}</span>
                  </>
                ) : (
                  <>
                    {(msg.content || msg.isStreaming) && (
                    <div className="bubble bubble-assistant">
                      {msg.isStreaming && !msg.content ? (
                        <span className="chat-typing-indicator">
                          <span className="chat-typing-dot" />
                          <span className="chat-typing-dot" />
                          <span className="chat-typing-dot" />
                        </span>
                      ) : (
                        <>
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                          {msg.isStreaming && <span className="chat-cursor" />}
                        </>
                      )}
                    </div>
                    )}
                    {msg.citations.length > 0 && (
                      <div className="cites">
                        {msg.citations.map((c, i) => (
                          <span key={i} className="cite">
                            <span className="src">{c.source_title}</span>
                            <span className="tspan">
                              {Math.floor(c.timestamp_start / 60)}:
                              {String(c.timestamp_start % 60).padStart(2, "0")}
                              –
                              {Math.floor(c.timestamp_end / 60)}:
                              {String(c.timestamp_end % 60).padStart(2, "0")}
                            </span>
                          </span>
                        ))}
                      </div>
                    )}
                    {msg.toolCall && (
                      <div className="toolcall">
                        <span className="ic"><FileText size={14} /></span>
                        <span>{t("chat.createdReport")} <strong>{msg.toolCall.result.name}</strong></span>
                        <span className="badge">{msg.toolCall.result.type.toUpperCase().replace(/_/g, " ")}</span>
                      </div>
                    )}
                    {msg.error && (
                      <div className="interrupted">
                        <span className="ic"><AlertCircle size={14} /></span>
                        <span>{msg.error}</span>
                        <button type="button" onClick={handleRetry} className="retry">
                          <RotateCcw size={12} />{t("chat.retry")}
                        </button>
                      </div>
                    )}
                    {!msg.isStreaming && !msg.error && (
                      <span className="ts">{msg.timestamp}</span>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Scroll-to-bottom button */}
        {showScrollButton && (
          <button
            type="button"
            onClick={scrollToBottom}
            aria-label={t("chat.scrollToBottom")}
            className="stb absolute bottom-14 left-1/2 flex h-8 w-8 -translate-x-1/2 items-center justify-center rounded-full border border-border bg-white shadow-lg transition hover:-translate-x-1/2 hover:-translate-y-px"
          >
            <ChevronDown size={16} className="text-ink" />
          </button>
        )}
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-border bg-white/55 px-3.5 py-5">
        <div className="relative">
          <div className={`input-wrap${!hasSources || isStreaming ? " disabled" : ""}`}>
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                autoResize(e.target);
              }}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={!hasSources || isStreaming}
              rows={1}
              className="w-full resize-none border-0 bg-transparent py-3 text-sm text-ink placeholder:text-muted focus:outline-none disabled:cursor-not-allowed disabled:text-muted"
            />

            <button
              type="button"
              onClick={isStreaming ? handleStop : () => void handleSend()}
              disabled={!hasSources || (!isStreaming && !inputValue.trim())}
              aria-label={isStreaming ? t("chat.stop") : t("chat.send")}
              className={`absolute bottom-1.5 right-1.5 flex items-center justify-center rounded-full text-white transition ${
                isStreaming
                  ? "bg-pink hover:brightness-110"
                  : "bg-blue hover:brightness-105 disabled:bg-border disabled:cursor-not-allowed"
              }`}
              style={{ width: "30px", height: "30px" }}
            >
              {isStreaming ? <Square size={12} fill="currentColor" /> : <SendHorizontal size={14} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
