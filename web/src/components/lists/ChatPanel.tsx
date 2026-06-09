import React, { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { useAutoScroll } from "@/components/lists/hooks/useAutoScroll";
import { useSSEStream } from "@/components/lists/hooks/useSSEStream";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Root, Element, Text, RootContent } from "hast";
import {
  AlertCircle,
  ChevronDown,
  Code,
  MessageSquare,
  MessageSquareOff,
  RotateCcw,
  SendHorizontal,
  Square,
  Trash2,
} from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { useJobActivity } from "@/components/jobs/JobActivityProvider";
import { useConversationHistory, type MessageUI } from "@/components/lists/hooks/useConversationHistory";
import { ToolLedger } from "@/components/lists/ToolLedger";
import { PulseRing } from "@/components/ui/PulseRing";
import { DebugDrawer } from "@/components/debug/DebugDrawer";
import type { Source } from "@/lib/types";
import { api } from "@/lib/api";
import { useDebugDump } from "@/lib/hooks/useDebugDump";
import { usePendingDeletions } from "@/lib/hooks/usePendingDeletions";
import { TEST_IDS } from "@/lib/test-ids";
import {
  autoResize,
  formatSubtitle,
  getErrorLabel,
  type ContentBlock,
} from "@/lib/chat-utils";

type OpenSourceOpts = {
  highlightChunks?: string[];
  sectionId?: string;
  timestampStart?: number;
};

function CitationChip({
  index,
  sourceId,
  chunkIds,
  sectionId,
  timestampStart,
  sources,
  onOpenSource,
}: {
  index: number;
  sourceId: string;
  chunkIds: string[];
  sectionId?: string;
  timestampStart?: number;
  sources: Source[];
  onOpenSource?: (source: Source, opts?: OpenSourceOpts) => void;
}) {
  const { t } = useLanguage();
  const source = sources.find((s) => s.id === sourceId);
  if (!source) {
    return (
      <span
        className="text-2xs text-muted cursor-not-allowed opacity-60"
        data-testid={TEST_IDS.citeMissing}
        title={t("chat.citationMissing")}
      >
        [{index}]
      </span>
    );
  }
  return (
    <span className="group/cite relative inline">
      <button
        type="button"
        onClick={() => onOpenSource?.(source, {
          highlightChunks: chunkIds,
          sectionId,
          timestampStart,
        })}
        data-testid={TEST_IDS.citeChip}
        className="mx-px border-0 bg-transparent p-0 text-2xs font-semibold text-blue cursor-pointer hover:underline focus-visible:rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue focus-visible:outline-offset-1"
      >
        [{index}]
      </button>
      <span
        data-testid={TEST_IDS.citeChipTooltip}
        className="pointer-events-none absolute bottom-full left-1/2 z-50 hidden w-max max-w-64 -translate-x-1/2 -translate-y-1.5 rounded-md bg-ink px-2 py-1 text-xs leading-snug text-white break-words group-hover/cite:block group-focus-within/cite:block"
      >
        {source.title}
      </span>
    </span>
  );
}


export const CITE_TOKEN_RE = /​⁣CITE(\d+)⁣​/;

function makeCiteToken(idx: number): string {
  // U+200B (ZWSP) + U+2063 (invisible separator) — both zero-width,
  // survive markdown parsing as inline text, won't be stripped by formatters.
  return `​⁣CITE${idx}⁣​`;
}

type CiteData = {
  index: number;
  source_id: string;
  chunk_ids: string[];
  section_id?: string;
  timestamp_start?: number;
  sources: Source[];
  onOpenSource?: (source: Source, opts?: OpenSourceOpts) => void;
};

function CiteEl(props: Record<string, any>) {
  const cite: CiteData | undefined = props["_cite"] as any;
  if (!cite) {
    console.warn("CiteEl: missing _cite prop — possible citation index mismatch");
    return null;
  }
  return (
    <CitationChip
      index={cite.index}
      sourceId={cite.source_id}
      chunkIds={cite.chunk_ids}
      sectionId={cite.section_id}
      timestampStart={cite.timestamp_start}
      sources={cite.sources}
      onOpenSource={cite.onOpenSource}
    />
  );
}

const MARKDOWN_COMPONENTS: Record<string, any> = {
  p: ({ children }: any) => <>{children}</>,
  "citation-el": CiteEl,
};

function makeRehypeCitePlugin(citations: CiteData[]) {
  // Attacher called by unified.use(); returns the actual transformer.
  return function rehypeCiteTokens(): (tree: Root) => void {
    return function transform(tree: Root): void {
      walk(tree);

      function walk(node: Root | Element): void {
        if (!node.children) return;
        for (let i = node.children.length - 1; i >= 0; i--) {
          const child = node.children[i];
          if (child.type === "text") {
            if (!CITE_TOKEN_RE.test(child.value)) continue;
            const parts = child.value.split(CITE_TOKEN_RE);
            const replacements: (Text | Element)[] = [];
            for (let j = 0; j < parts.length; j++) {
              if (j % 2 === 0) {
                if (parts[j]) replacements.push({ type: "text", value: parts[j] });
              } else {
                const idx = Number(parts[j]);
                if (!citations[idx]) {
                  console.warn("rehypeCiteTokens: cite token out of bounds", parts[j], citations.length);
                  continue;
                }
                replacements.push({
                  type: "element",
                  tagName: "citation-el",
                  properties: { _cite: citations[idx] as any },
                  children: [],
                });
              }
            }
            node.children.splice(i, 1, ...replacements as RootContent[]);
          } else if (child.type === "element") {
            walk(child);
          }
        }
      }
    };
  };
}

function renderParagraphs(
  contentBlocks: ContentBlock[],
  sources: Source[],
  onOpenSource?: (source: Source, opts?: OpenSourceOpts) => void,
  isStreaming?: boolean,
) {
  // Split into paragraphs on paragraph_break
  const paragraphs: Array<Array<ContentBlock>> = [[]];
  const last = () => paragraphs[paragraphs.length - 1];
  for (const block of contentBlocks) {
    if (block.type === "paragraph_break") {
      if (last().length > 0) {
        paragraphs.push([]);
      }
    } else {
      last().push(block);
    }
  }

  // Post-merge fold: citation-only trailing paragraphs attach to previous
  for (let i = paragraphs.length - 1; i > 0; i--) {
    if (paragraphs[i].length > 0 && paragraphs[i].every((b) => b.type === "citation")) {
      paragraphs[i - 1].push(...paragraphs[i]);
      paragraphs[i] = [];
    }
  }

  return (
    <>
      {paragraphs.map((para, pi) => {
        if (para.length === 0) return null;

        const citations: CiteData[] = [];
        let merged = "";
        for (const block of para) {
          if (block.type === "text") {
            merged += block.text;
          } else if (block.type === "citation") {
            merged += makeCiteToken(citations.length);
            citations.push({
              index: block.index,
              source_id: block.source_id,
              chunk_ids: block.chunk_ids,
              section_id: block.section_id,
              timestamp_start: block.timestamp_start,
              sources,
              onOpenSource,
            });
          }
        }

        return (
          <div key={pi} className="citation-paragraph">
            <ReactMarkdown
              components={MARKDOWN_COMPONENTS}
              rehypePlugins={[makeRehypeCitePlugin(citations)]}
              remarkPlugins={[remarkGfm]}
            >
              {merged}
            </ReactMarkdown>
          </div>
        );
      })}
      {isStreaming && (
        <span className="inline-block w-0.5 h-3.5 bg-blue align-text-bottom ml-0.5 chat-cursor-blink" />
      )}
    </>
  );
}

function AssistantBubble({
  children,
  showDebugButton,
  onShowDebug,
}: {
  children: ReactNode;
  showDebugButton?: boolean;
  onShowDebug?: () => void;
}) {
  return (
    <div
      data-testid={TEST_IDS.bubbleAssistant}
      className="bubble relative rounded-2xl rounded-bl-md border border-border bg-white/70"
    >
      {children}
      {showDebugButton && onShowDebug && (
        <button
          type="button"
          onClick={onShowDebug}
          title="View LLM context (debug)"
          aria-label="View LLM context (debug)"
          className="absolute bottom-2 right-2 flex h-6 w-6 items-center justify-center rounded-md border border-border bg-white text-muted shadow-sm hover:text-blue"
        >
          <Code size={12} />
        </button>
      )}
    </div>
  );
}

interface ChatPanelProps {
  selectedSourceIds: string[];
  sources: Source[];
  listId: string;
  onOpenSource?: (source: Source, opts?: OpenSourceOpts) => void;
}

export function ChatPanel({
  selectedSourceIds,
  sources,
  listId,
  onOpenSource,
}: ChatPanelProps) {
  const { t } = useLanguage();
  const { trackJobs } = useJobActivity();
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<MessageUI[]>([]);
  const [showClearPopover, setShowClearPopover] = useState(false);
  const [debugPrompts, setDebugPrompts] = useState(false);
  const [debugMsgId, setDebugMsgId] = useState<string | null>(null);
  const { dump: debugDump, loading: debugLoading, notFound: debugNotFound, reset: resetDebugDump } = useDebugDump(debugMsgId);

  // Wrap setDebugMsgId with a synchronous reset so the drawer doesn't
  // flash stale data on a re-open (React batches both updates so the next
  // render sees dump=null, msgId=new).
  const openDebug = (id: string) => {
    resetDebugDump();
    setDebugMsgId(id);
  };
  const { isPending, run } = usePendingDeletions();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getConfig()
      .then((c) => {
        if (!cancelled) setDebugPrompts(c?.rag?.debug_prompts ?? false);
      })
      .catch(() => {
        // debug prompts is opt-in debug-only; missing config just leaves it off
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!debugMsgId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDebugMsgId(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [debugMsgId]);

  useEffect(() => {
    if (debugNotFound && debugMsgId) {
      console.warn("debug dump not found", debugMsgId);
    }
  }, [debugNotFound, debugMsgId]);

  const hasSources = selectedSourceIds.length > 0;
  const { messages: historyMessages, isLoadingHistory, loadError, activeStreamMessageId } = useConversationHistory(listId, hasSources, t("chat.interrupted"), t("chat.stopped"));

  const { sendMessage, stopStreaming, retryMessage, reattach, isStreaming } = useSSEStream({
    listId,
    selectedSourceIds,
    messages,
    setMessages,
    trackJobs,
    interruptedLabel: t("chat.interrupted"),
    stoppedLabel: t("chat.stopped"),
  });

  const reattachedRef = useRef<string | null>(null);

  const { isAtBottom, messageListRef, scrollToBottom } = useAutoScroll({
    isLoadingHistory,
    messages,
  });
  const showScrollButton = !isAtBottom;

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

  useEffect(() => {
    if (historyMessages.length > 0) {
      setMessages(historyMessages);
    }
  }, [historyMessages]);

  useEffect(() => {
    if (
      activeStreamMessageId &&
      !isStreaming &&
      reattachedRef.current !== activeStreamMessageId
    ) {
      reattachedRef.current = activeStreamMessageId;
      void reattach(activeStreamMessageId);
    }
  }, [activeStreamMessageId, isStreaming, reattach]);

  function handleSend() {
    const text = inputValue.trim();
    if (!text || !hasSources || isStreaming) return;
    setInputValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.overflowY = "hidden";
    }
    void sendMessage(text);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      isStreaming ? stopStreaming() : handleSend();
    }
  }

  async function handleClearConfirm() {
    setShowClearPopover(false);
    try {
      await run(listId, () => api.deleteConversation(listId));
    } catch {
      // non-fatal — keep messages, surface nothing
      return;
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
              disabled={!hasConversation || isPending(listId)}
              aria-label={t("chat.clearConfirm.title")}
              className="flex h-7 w-7 items-center justify-center rounded-lg text-muted transition hover:bg-border disabled:cursor-not-allowed disabled:opacity-30 hover:disabled:bg-transparent"
            >
              <Trash2 size={14} />
            </button>

            {showClearPopover && (
              <div className="absolute right-0 top-9 z-30 w-60 rounded-xl border border-border bg-white p-3.5 shadow-lg">
                <span
                  aria-hidden="true"
                  className="absolute -top-1.5 right-4.5 size-2.5 rotate-45 border-l border-t border-border bg-white"
                />
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
                    disabled={isPending(listId)}
                    className="rounded-full bg-ink px-3 py-1.5 text-xs font-medium text-white transition hover:brightness-110 disabled:opacity-60"
                  >
                    {isPending(listId) ? t("common.deleting") : t("chat.clearConfirm.clear")}
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

      {/* Message list + scroll-to-bottom wrapper */}
      <div className="relative min-h-0 flex-1">
        <div
          ref={messageListRef}
          role="region"
          aria-label={t("chat.header.title")}
          className={`absolute inset-0 flex flex-col gap-3.5 overflow-y-auto px-4.5 py-4 ${showClearPopover ? "opacity-50" : ""}`}
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
        ) : loadError && !hasConversation ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-surface text-muted">
              <AlertCircle size={26} />
            </div>
            <p className="m-0 max-w-xs text-sm text-muted">{loadError}</p>
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
            {messages.map((msg) => {
              const showLedger =
                !msg.isStreaming ||
                msg.contentBlocks.length > 0 ||
                msg.content ||
                msg.pendingRagCalls.length > 0 ||
                (msg.rag?.calls?.length ?? 0) > 0;
              return (
              <div
                key={msg.id}
                className={`flex max-w-2xl flex-col gap-1.5 ${
                  msg.role === "user" ? "self-end items-end" : "self-start items-start"
                }`}
              >
                {msg.role === "user" ? (
                  <>
                    <div
                      data-testid={TEST_IDS.bubbleUser}
                      className="bubble rounded-2xl rounded-br-md border border-sky-35 bg-sky/10"
                    >
                      {msg.content}
                    </div>
                    <span className="text-2xs text-muted font-mono px-1">{msg.timestamp}</span>
                  </>
                ) : (
                  <>
                    {showLedger ? (
                      <ToolLedger
                        ragCalls={msg.rag?.calls ?? []}
                        pendingRagCalls={msg.pendingRagCalls}
                        streaming={msg.isStreaming}
                      />
                    ) : null}
                    {msg.isStreaming && !msg.content && !msg.contentBlocks.length ? (
                      <PulseRing />
                    ) : msg.contentBlocks.length > 0 ? (
                      <AssistantBubble
                        showDebugButton={debugPrompts && msg.hasDump}
                        onShowDebug={() => openDebug(msg.id)}
                      >
                        {renderParagraphs(msg.contentBlocks, sources, onOpenSource, msg.isStreaming)}
                      </AssistantBubble>
                    ) : msg.content ? (
                      <AssistantBubble
                        showDebugButton={debugPrompts && msg.hasDump}
                        onShowDebug={() => openDebug(msg.id)}
                      >
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </AssistantBubble>
                    ) : null}
                    {msg.error && (
                      <div className="interrupted">
                        <span className="ic"><AlertCircle size={14} /></span>
                        <span>{getErrorLabel(msg.error, t)}</span>
                        <button
                          type="button"
                          onClick={() => retryMessage(msg.id)}
                          className="ml-auto inline-flex cursor-pointer items-center gap-1 border-0 bg-transparent p-0 text-xs font-medium text-blue hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue focus-visible:outline-offset-2"
                        >
                          <RotateCcw size={12} />{t("chat.retry")}
                        </button>
                      </div>
                    )}
                    {!msg.isStreaming && !msg.error && (
                      <span className="text-2xs text-muted font-mono px-1">{msg.timestamp}</span>
                    )}
                  </>
                )}
              </div>
            ); })}
          </div>
        )}
        </div>

        {/* Scroll-to-bottom button */}
        {showScrollButton && (
          <button
            type="button"
            onClick={scrollToBottom}
            aria-label={t("chat.scrollToBottom")}
            className="absolute inset-x-0 mx-auto bottom-4 z-20 inline-flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border border-border bg-white text-ink shadow-md transition duration-150 hover:-translate-y-px focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue focus-visible:outline-offset-2"
          >
            <ChevronDown size={16} />
          </button>
        )}
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-border bg-white/55 px-3.5 py-5">
        <div className="relative">
          <div
            className={`relative rounded-2xl border py-2.5 pl-3.5 pr-11 transition-all duration-150 ${
              !hasSources || isStreaming
                ? "cursor-not-allowed border-slate-200/50 bg-slate-100/70"
                : "border-border bg-white/90 focus-within:border-blue/25 focus-within:ring-2 focus-within:ring-sky-35"
            }`}
          >
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
              onClick={isStreaming ? stopStreaming : handleSend}
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

      {debugMsgId && !debugLoading && debugDump ? (
        <DebugDrawer
          messageId={debugMsgId}
          dump={debugDump}
          onClose={() => setDebugMsgId(null)}
        />
      ) : null}
    </div>
  );
}
