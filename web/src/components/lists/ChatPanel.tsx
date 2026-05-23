import React, { useEffect, useMemo, useRef, useState } from "react";

import { useAutoScroll } from "@/components/lists/hooks/useAutoScroll";
import { useSSEStream } from "@/components/lists/hooks/useSSEStream";
import ReactMarkdown from "react-markdown";
import type { Root, Element, Text, RootContent } from "hast";
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
import { useConversationHistory, type MessageUI } from "@/components/lists/hooks/useConversationHistory";
import { RetrievalLedger } from "@/components/lists/RetrievalLedger";
import { PulseRing } from "@/components/ui/PulseRing";
import type { Source } from "@/lib/types";
import { api } from "@/lib/api";
import {
  autoResize,
  formatSubtitle,
  getErrorLabel,
  type ContentBlock,
} from "@/lib/chat-utils";

function CitationChip({
  index,
  sourceId,
  chunkIds,
  sources,
  onOpenSource,
}: {
  index: number;
  sourceId: string;
  chunkIds: string[];
  sources: Source[];
  onOpenSource?: (source: Source, opts?: { highlightChunks?: string[] }) => void;
}) {
  const { t } = useLanguage();
  const source = sources.find((s) => s.id === sourceId);
  if (!source) {
    return (
      <span className="cite-missing" title={t("chat.citationMissing")}>
        [{index}]
      </span>
    );
  }
  return (
    <span className="cite-chip-wrap">
      <button type="button" className="cite-chip" onClick={() => onOpenSource?.(source, { highlightChunks: chunkIds })}>
        [{index}]
      </button>
      <span className="cite-chip-tooltip">{source.title}</span>
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
  sources: Source[];
  onOpenSource?: (source: Source, opts?: { highlightChunks?: string[] }) => void;
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
  onOpenSource?: (source: Source, opts?: { highlightChunks?: string[] }) => void,
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
            >
              {merged}
            </ReactMarkdown>
          </div>
        );
      })}
      {isStreaming && <span className="chat-cursor" />}
    </>
  );
}

interface ChatPanelProps {
  selectedSourceIds: string[];
  sources: Source[];
  listId: string;
  onOpenSource?: (source: Source, opts?: { highlightChunks?: string[] }) => void;
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
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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

  const { showScrollButton, messageListRef, scrollToBottom } = useAutoScroll({
    isLoadingHistory,
    isStreaming,
    messages,
  });

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
                msg.pendingMetadataCalls.length > 0 ||
                (msg.rag?.calls?.length ?? 0) > 0;
              return (
              <div key={msg.id} className={`msg ${msg.role}`}>
                {msg.role === "user" ? (
                  <>
                    <div className="bubble bubble-user">{msg.content}</div>
                    <span className="ts">{msg.timestamp}</span>
                  </>
                ) : (
                  <>
                    {showLedger ? (
                      <RetrievalLedger
                        calls={msg.rag?.calls ?? []}
                        pendingRetrieve={msg.pendingRagCalls}
                        pendingMetadata={msg.pendingMetadataCalls}
                        streaming={msg.isStreaming}
                      />
                    ) : null}
                    {msg.isStreaming && !msg.content && !msg.contentBlocks.length ? (
                      <PulseRing />
                    ) : msg.contentBlocks.length > 0 ? (
                      <div className="bubble bubble-assistant">
                        {renderParagraphs(msg.contentBlocks, sources, onOpenSource, msg.isStreaming)}
                      </div>
                    ) : msg.content ? (
                      <div className="bubble bubble-assistant">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    ) : null}
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
                        <span>{getErrorLabel(msg.error, t)}</span>
                        <button type="button" onClick={() => retryMessage(msg.id)} className="retry">
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
            ); })}
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

    </div>
  );
}
