import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, FileText, X } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import type { Artifact, Source } from "@/lib/types";
import type { MindMapAskInChat, OpenSourceOpts } from "@/lib/chat-utils";
import { TEST_IDS } from "@/lib/test-ids";
import { MindMapBlock } from "./MindMapBlock";

interface ArtifactViewerProps {
  artifact: Artifact;
  sources?: Source[];
  onAskInChatFromMindmap?: MindMapAskInChat;
  onOpenSource?: (source: Source, opts?: OpenSourceOpts) => void;
}

const MARKDOWN_COMPONENTS = {
  pre({ children }: { children?: ReactNode }) {
    return (
      <pre className="overflow-x-auto rounded-lg bg-border/30 p-3 text-xs">
        {children}
      </pre>
    );
  },
};

export function ArtifactViewer({ artifact, sources, onAskInChatFromMindmap, onOpenSource }: ArtifactViewerProps) {
  const { t } = useLanguage();
  const [content, setContent] = useState<string | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const sourcesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const result = await api.getArtifactContent(artifact.id);
        if (cancelled || !result) return;
        setContent(result.content);
      } catch {
        // Non-critical: content stays null
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [artifact.id]);

  // Close the sources popover on outside click / Escape.
  useEffect(() => {
    if (!sourcesOpen) return;
    function onDocMouseDown(e: MouseEvent) {
      if (sourcesRef.current && !sourcesRef.current.contains(e.target as Node)) {
        setSourcesOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setSourcesOpen(false);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [sourcesOpen]);

  function handleCopy() {
    if (!content) return;
    void navigator.clipboard.writeText(content);
  }

  const sourceCount = artifact.source_ids.length;
  const isMindMap = artifact.type === "mind_map";
  const sourcesById = useMemo(
    () => new Map((sources ?? []).map((s) => [s.id, s])),
    [sources],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate font-serif text-base font-bold text-ink">{artifact.name}</p>
          {isMindMap ? (
            sourceCount > 0 && (
              <div ref={sourcesRef} className="relative mt-1.5">
                <button
                  type="button"
                  onClick={() => setSourcesOpen((v) => !v)}
                  aria-expanded={sourcesOpen}
                  aria-haspopup="dialog"
                  className="inline-flex items-center rounded-full border border-border bg-white/80 px-3 py-1 text-xs font-medium text-ink transition hover:bg-white"
                >
                  {t("lab.artifactViewer.viewSources", { count: sourceCount })}
                </button>
                {sourcesOpen && (
                  <div
                    role="dialog"
                    aria-label={t("lab.artifactViewer.sources")}
                    className="absolute left-0 top-9 z-30 w-72 max-h-72 overflow-hidden rounded-xl border border-border bg-white shadow-lg"
                  >
                    <div className="flex items-center justify-between border-b border-border px-3 py-2">
                      <span className="text-sm font-semibold text-ink">
                        {t("lab.artifactViewer.sources")}
                      </span>
                      <button
                        type="button"
                        onClick={() => setSourcesOpen(false)}
                        aria-label={t("lab.artifactViewer.close")}
                        className="flex h-6 w-6 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
                      >
                        <X size={12} />
                      </button>
                    </div>
                    {/* Render exactly `sourceCount` rows so the pill count
                        matches the visible list. Missing ids (source was
                        deleted after the mindmap was generated) show as a
                        grayed placeholder row. */}
                    <div className="max-h-56 overflow-y-auto py-1">
                      {artifact.source_ids.map((id) => {
                        const source = sourcesById.get(id);
                        if (!source) {
                          return (
                            <div
                              key={id}
                              data-testid={TEST_IDS.sourceRowDeleted}
                              className="flex items-center gap-2 px-3 py-1.5 text-xs text-muted opacity-60"
                            >
                              <FileText size={14} className="shrink-0" />
                              <span className="truncate line-through">
                                {t("lab.artifactViewer.deletedSource")}
                              </span>
                            </div>
                          );
                        }
                        return (
                          <button
                            key={id}
                            type="button"
                            data-testid={TEST_IDS.sourceRow}
                            onClick={() => {
                              onOpenSource?.(source);
                              setSourcesOpen(false);
                            }}
                            title={source.title}
                            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-ink transition hover:bg-border/40 focus-visible:bg-border/40 focus-visible:outline-none"
                          >
                            <FileText size={14} className="shrink-0 text-muted" />
                            <span className="truncate">{source.title}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )
          ) : (
            <p className="m-0 mt-0.5 text-xs text-muted">
              {t(sourceCount === 1 ? "lab.artifactViewer.basedOnSingular" : "lab.artifactViewer.basedOnPlural", { count: sourceCount })}
            </p>
          )}
        </div>
        {!isMindMap && (
          <button
            type="button"
            aria-label="Copy markdown"
            disabled={!content}
            onClick={handleCopy}
            className="flex h-7 w-7 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink disabled:opacity-40"
          >
            <Copy size={14} />
          </button>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {content ? (
          isMindMap ? (
            <MindMapBlock
              content={content}
              onAskInChat={
                onAskInChatFromMindmap
                  ? (topic, parent, evidence) =>
                      onAskInChatFromMindmap(topic, parent, artifact.source_ids, evidence)
                  : undefined
              }
            />
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
                {content}
              </ReactMarkdown>
            </div>
          )
        ) : (
          <p className="text-sm text-muted">{t("lab.artifactViewer.loading")}</p>
        )}
      </div>
    </div>
  );
}
