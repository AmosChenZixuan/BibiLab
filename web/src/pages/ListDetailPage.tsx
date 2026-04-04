/**
 * ListDetailPage — redesign per docs/specs/2026-04-03-list-detail-redesign.md
 *
 * Slice 1: Page shell, resize, skeleton panels.
 * Slice 2: Navbar title portal + rename.
 * Slice 3: Source list, URL bar, ingest, job rows, context menu.
 * Chat and Lab panels render as skeletons. Sources panel shows its header and
 * collapse toggle with an empty body. No API wiring yet.
 */

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
import { useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import {
  MdChevronLeft,
  MdChevronRight,
  MdMoreVert,
  MdRefresh,
  MdDeleteOutline,
  MdClose,
  MdArrowForward,
  MdErrorOutline,
  MdExpandLess,
  MdExpandMore,
} from "react-icons/md";

import { ContextMenu } from "../components/ui/ContextMenu";
import { useJobActivity } from "../components/jobs/JobActivityProvider";
import { api, toErrorMessage } from "../lib/api";
import type { NoteContent, Source } from "../lib/types";

// ─── Constants ────────────────────────────────────────────────────────────────

const MIN_PANEL = 280;
const COLLAPSED_PANEL = 48;
const RESIZER_SIZE = 16;

// ─── Panel resize manager ─────────────────────────────────────────────────────

type ActiveResizer = "left" | "right" | null;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function getResizableWorkspaceWidth(containerWidth: number) {
  return Math.max(0, containerWidth - RESIZER_SIZE * 2);
}

function getContainerContentWidth(
  containerWidth: number,
  paddingLeft: number,
  paddingRight: number,
) {
  return Math.max(0, containerWidth - paddingLeft - paddingRight);
}

function clampSourcesWidth(
  nextWidth: number,
  workspaceWidth: number,
  labWidth: number,
  sourcesMinWidth: number,
) {
  const maxWidth = Math.max(sourcesMinWidth, workspaceWidth - labWidth - MIN_PANEL);
  return clamp(nextWidth, sourcesMinWidth, maxWidth);
}

function clampLabWidth(nextWidth: number, workspaceWidth: number, sourcesWidth: number) {
  const maxWidth = Math.max(MIN_PANEL, workspaceWidth - sourcesWidth - MIN_PANEL);
  return clamp(nextWidth, MIN_PANEL, maxWidth);
}

function initialEqualPanelW() {
  return Math.floor((window.innerWidth - 32 - RESIZER_SIZE * 2) / 3);
}

function usePanelResize(
  containerRef: RefObject<HTMLDivElement | null>,
  sourcesCollapsed: boolean,
) {
  const [sourcesW, setSourcesW] = useState(initialEqualPanelW);
  const [labW, setLabW] = useState(initialEqualPanelW);
  const [containerContentWidth, setContainerContentWidth] = useState(0);

  const active = useRef<ActiveResizer>(null);
  const startX = useRef(0);
  const startSourcesW = useRef(288);
  const startLabW = useRef(288);

  const sourcesWRef = useRef(sourcesW);
  const labWRef = useRef(labW);
  sourcesWRef.current = sourcesW;
  labWRef.current = labW;

  const onMouseDownLeft = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    active.current = "left";
    startX.current = e.clientX;
    startSourcesW.current = sourcesWRef.current;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const onMouseDownRight = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    active.current = "right";
    startX.current = e.clientX;
    startLabW.current = labWRef.current;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const element = container;

    function measure() {
      const styles = window.getComputedStyle(element);
      setContainerContentWidth(
        getContainerContentWidth(
          element.clientWidth,
          Number.parseFloat(styles.paddingLeft) || 0,
          Number.parseFloat(styles.paddingRight) || 0,
        ),
      );
    }

    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(element);
    window.addEventListener("resize", measure);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [containerRef]);

  useEffect(() => {
    const workspaceWidth = getResizableWorkspaceWidth(containerContentWidth);
    if (!workspaceWidth) return;

    const sourcesMinWidth = sourcesCollapsed ? COLLAPSED_PANEL : MIN_PANEL;
    const nextSourcesW = clampSourcesWidth(sourcesW, workspaceWidth, labW, sourcesMinWidth);
    const nextLabW = clampLabWidth(labW, workspaceWidth, nextSourcesW);

    if (nextSourcesW !== sourcesW) {
      setSourcesW(nextSourcesW);
    }

    if (nextLabW !== labW) {
      setLabW(nextLabW);
    }
  }, [containerContentWidth, labW, sourcesCollapsed, sourcesW]);

  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!active.current) return;
      const delta = e.clientX - startX.current;
      const workspaceWidth = getResizableWorkspaceWidth(containerContentWidth);
      if (!workspaceWidth) return;
      const sourcesMinWidth = sourcesCollapsed ? COLLAPSED_PANEL : MIN_PANEL;

      if (active.current === "left") {
        setSourcesW(
          clampSourcesWidth(
            startSourcesW.current + delta,
            workspaceWidth,
            labWRef.current,
            sourcesMinWidth,
          ),
        );
      } else {
        setLabW(
          clampLabWidth(
            startLabW.current - delta,
            workspaceWidth,
            sourcesCollapsed ? COLLAPSED_PANEL : sourcesWRef.current,
          ),
        );
      }
    }

    function onUp() {
      if (!active.current) return;
      active.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [containerContentWidth, sourcesCollapsed]);

  const workspaceWidth = getResizableWorkspaceWidth(containerContentWidth);
  const sourcesWidth = sourcesCollapsed ? COLLAPSED_PANEL : sourcesW;
  const chatW = Math.max(MIN_PANEL, workspaceWidth - sourcesWidth - labW);

  return { sourcesW, labW, chatW, onMouseDownLeft, onMouseDownRight };
}

// ─── Resizer handle ───────────────────────────────────────────────────────────

function Resizer({ onMouseDown }: { onMouseDown: (e: React.MouseEvent) => void }) {
  return (
    <div
      className="shrink-0 cursor-col-resize self-stretch"
      style={{ width: `${RESIZER_SIZE}px` }}
      onMouseDown={onMouseDown}
    />
  );
}

// ─── Skeleton panel ──────────────────────────────────────────────────────────

function SkeletonPanel({ title, note }: { title: string; note: string }) {
  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-border px-5 py-4">
        <h2 className="m-0 font-serif text-lg text-ink">{title}</h2>
      </div>
      <div className="flex flex-1 flex-col items-center justify-center gap-4 px-6 py-8">
        <div className="w-full space-y-2.5">
          <div className="h-2.5 w-5/6 rounded-full bg-linear-to-r from-pink/12 to-sky/12" />
          <div className="h-2.5 rounded-full bg-linear-to-r from-pink/12 to-sky/12" />
          <div className="h-2.5 w-2/3 rounded-full bg-linear-to-r from-pink/12 to-sky/12" />
        </div>
        <p className="m-0 text-center text-sm text-muted/80">{note}</p>
      </div>
    </div>
  );
}

// ─── Navbar title portal ──────────────────────────────────────────────────────

function NavbarTitle({
  name,
  onCommit,
}: {
  name: string;
  onCommit: (newName: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);
  const inputRef = useRef<HTMLInputElement>(null);
  const navEl = document.querySelector("nav");

  // Sync draft when name changes externally
  useEffect(() => {
    if (!editing) {
      setDraft(name);
    }
  }, [name, editing]);

  useEffect(() => {
    if (editing) {
      inputRef.current?.select();
    }
  }, [editing]);

  function commit() {
    const trimmed = draft.trim();
    const next = trimmed || name;
    setDraft(next);
    setEditing(false);
    if (trimmed && trimmed !== name) {
      void onCommit(trimmed);
    }
  }

  if (!navEl) return null;

  return createPortal(
    // Positioned right of the logo (~left-24 at px-4 spacing)
    <div className="absolute left-24 top-1/2 -translate-y-1/2 flex items-center">
      {editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") { setDraft(name); setEditing(false); }
          }}
          className="w-64 rounded-sm border border-blue/30 bg-sky/6 p-1 text-lg font-medium text-ink outline-none focus:border-blue/50 focus:bg-white transition"
          autoFocus
        />
      ) : (
        <span
          role="button"
          tabIndex={0}
          onClick={() => { setDraft(name); setEditing(true); }}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              setDraft(name);
              setEditing(true);
            }
          }}
          className="truncate cursor-text rounded-sm border border-transparent px-1 py-0.5 text-lg font-medium text-ink leading-normal transition hover:border-blue/30"
        >
          {name}
        </span>
      )}
    </div>,
    navEl,
  );
}

// ─── Pipeline stages ──────────────────────────────────────────────────────────

// 6-segment track per spec: queued → downloading → transcribing → extracting → writing → done
const PIPELINE_STAGES = [
  "queued",
  "downloading",
  "transcribing",
  "extracting",
  "writing",
  "done",
] as const;
type PipelineStage = (typeof PIPELINE_STAGES)[number];

const STAGE_LABELS: Record<PipelineStage, string> = {
  queued:       "Queued",
  downloading:  "Downloading",
  transcribing: "Transcribing",
  extracting:   "Extracting",
  writing:      "Writing",
  done:         "Done",
};

// ─── Source row ───────────────────────────────────────────────────────────────

function SourceRow({
  source,
  onOpen,
  onDelete,
  onRerun,
}: {
  source: Source;
  onOpen: () => void;
  onDelete: () => Promise<void>;
  onRerun: () => Promise<void>;
}) {
  return (
    <div className="group flex items-center gap-2 rounded-2xl border border-border bg-white/64 px-4 py-3 transition hover:bg-white hover:shadow-sm">
      <button
        type="button"
        aria-label={`Open ${source.title}`}
        className="min-w-0 flex-1 border-0 bg-transparent text-left"
        onClick={onOpen}
      >
        <p className="m-0 truncate text-sm font-medium text-ink">{source.title}</p>
        <p className="m-0 mt-0.5 text-xs text-muted">{source.platform}</p>
      </button>
      <ContextMenu
        items={[
          { label: "Re-run", icon: <MdRefresh />, onClick: onRerun },
          { label: "Delete", icon: <MdDeleteOutline />, onClick: onDelete, variant: "danger" },
        ]}
        trigger={({ toggle, triggerRef }) => (
          <button
            ref={triggerRef}
            type="button"
            aria-label="Source options"
            onClick={toggle}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted opacity-0 transition group-hover:opacity-100 hover:bg-border hover:text-ink"
          >
            <MdMoreVert size={16} />
          </button>
        )}
      />
    </div>
  );
}

// ─── Ingesting source row ─────────────────────────────────────────────────────

function IngestingSourceRow({
  stage,
  title,
  onDismiss,
}: {
  stage: string;
  title: string;
  onDismiss: () => void;
}) {
  const isFailed = stage === "failed" || stage === "needs_auth";
  const stageIdx = isFailed ? -1 : PIPELINE_STAGES.indexOf(stage as PipelineStage);
  const displayStage = isFailed ? "failed" : (PIPELINE_STAGES[Math.max(0, stageIdx)] ?? "downloading");

  if (isFailed) {
    return (
      <div className="flex items-start gap-3 rounded-2xl border border-pink/30 bg-pink/6 px-4 py-3">
        <MdErrorOutline size={16} className="mt-0.5 shrink-0 text-pink" />
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-sm font-medium text-ink">{title}</p>
          <p className="m-0 mt-0.5 text-xs text-pink/80">Failed during {displayStage}</p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
        >
          <MdClose size={14} />
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5 rounded-2xl border border-blue/20 bg-sky/6 px-4 py-3">
      <div className="flex items-center gap-2">
        {/* Animated dot */}
        <span className="relative flex h-2 w-2 shrink-0">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue/40" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-blue/70" />
        </span>
        <p className="m-0 min-w-0 flex-1 truncate text-sm font-medium text-ink">{title}</p>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Cancel ingestion"
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
        >
          <MdClose size={14} />
        </button>
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs text-blue/80">{STAGE_LABELS[displayStage as PipelineStage] ?? stage}…</span>
          <span className="text-xs tabular-nums text-muted/60">
            {Math.max(1, stageIdx + 1)} / {PIPELINE_STAGES.length}
          </span>
        </div>
        <div className="flex gap-0.5">
          {PIPELINE_STAGES.map((s, i) => (
            <div
              key={s}
              title={STAGE_LABELS[s]}
              className={`h-1 flex-1 rounded-full transition-colors duration-500 ${
                i < stageIdx
                  ? "bg-blue/60"
                  : i === stageIdx
                    ? "animate-pulse bg-blue/90"
                    : "bg-border"
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Note accordion ───────────────────────────────────────────────────────────

function NoteAccordion({ markdown }: { markdown: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-blue/25">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between gap-3 border-0 bg-transparent px-4 py-3.5 text-left transition hover:bg-sky/6"
      >
        <span className="text-sm font-semibold text-ink">Note</span>
        {expanded
          ? <MdExpandLess size={18} className="shrink-0 text-muted" />
          : <MdExpandMore size={18} className="shrink-0 text-muted" />}
      </button>
      {expanded && (
        <div className="border-t border-border px-4 py-4 space-y-2 text-sm text-muted [&_p]:text-sm [&_p]:text-muted [&_p]:leading-relaxed [&_p]:mb-2 [&_strong]:font-semibold [&_strong]:text-ink [&_h1]:text-sm [&_h1]:font-semibold [&_h1]:text-ink [&_h2]:text-xs [&_h2]:font-semibold [&_h2]:text-muted [&_h2]:uppercase [&_h2]:tracking-wider [&_ul]:pl-4 [&_ul]:space-y-1 [&_li]:text-sm [&_li]:text-muted [&_blockquote]:border-l-2 [&_blockquote]:border-blue/25 [&_blockquote]:pl-3 [&_blockquote]:italic [&_img]:max-w-full [&_img]:rounded-lg [&_img]:my-2">
          <ReactMarkdown>{markdown}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

// ─── Sources viewer mode ───────────────────────────────────────────────────────

function SourcesViewerMode({
  source,
  note,
  transcript,
  transcriptError,
  onClose,
}: {
  source: Source;
  note: NoteContent | null;
  transcript: string | null;
  transcriptError: string | null;
  onClose: () => void;
}) {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex shrink-0 items-start gap-3 border-b border-border px-4 py-4">
        <button
          type="button"
          onClick={onClose}
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
          aria-label="Close viewer"
        >
          <MdClose size={16} />
        </button>
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-sm font-medium text-ink">{source.title}</p>
          <p className="m-0 mt-0.5 text-xs text-muted">{source.platform}</p>
        </div>
      </div>

      {/* Doc viewer — scrollable */}
      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {note && <NoteAccordion markdown={note.markdown} />}

        {/* Transcript — auto-loaded, shown directly */}
        <div className="space-y-2">
          {transcriptError && (
            <p className="text-xs text-rose-700">{transcriptError}</p>
          )}
          {transcript && (
            <>
              <p className="text-xs font-semibold uppercase tracking-wider text-muted/70">Transcript</p>
              <pre className="p-1 m-0 whitespace-pre-wrap font-mono text-xs text-muted leading-relaxed">
                {transcript}
              </pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Sources list mode ─────────────────────────────────────────────────────────

function SourcesListMode({
  listId,
  sources,
  onOpenSource,
}: {
  listId: string;
  sources: Source[];
  onOpenSource: (source: Source) => void;
}) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { dismissJob, getJobs, trackJobs } = useJobActivity();
  const ingestJobs = getJobs("ingest", listId);
  const refreshedJobsRef = useRef<string[]>([]);

  // Auto-refresh sources when a job completes
  const [sourcesVersion, setSourcesVersion] = useState(0);
  const [currentSources, setCurrentSources] = useState(sources);

  // Sync external sources prop
  useEffect(() => {
    setCurrentSources(sources);
  }, [sources, sourcesVersion]);

  // When a job flips to done, refresh sources and dismiss
  useEffect(() => {
    const completed = ingestJobs.filter(
      (j) => j.isTerminal && j.job.status === "done" && !refreshedJobsRef.current.includes(j.job.id),
    );
    if (completed.length === 0) return;

    let cancelled = false;
    async function refresh() {
      try {
        const next = await api.listSources(listId);
        if (cancelled) return;
        setCurrentSources(next);
        for (const { job } of completed) {
          refreshedJobsRef.current.push(job.id);
          await dismissJob(job.id);
        }
      } catch {
        // Non-critical: leave jobs in terminal state
      }
    }
    void refresh();
    return () => {
      cancelled = true;
    };
  }, [ingestJobs, listId]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) return;
    setUrl("");
    setError(null);
    try {
      const result = await api.ingestUrl(listId, trimmed, false);
      trackJobs(
        result.queued.map((id) => ({
          id,
          producer: "ingest" as const,
          label: trimmed,
          contextKey: listId,
        })),
      );
    } catch (err) {
      setUrl(trimmed);
      setError(err instanceof Error ? err.message : "Failed to submit URL");
    }
  }

  async function handleDelete(source: Source) {
    await api.deleteSource(listId, source.video_id);
    setCurrentSources((prev) => prev.filter((s) => s.video_id !== source.video_id));
  }

  async function handleRerun(source: Source) {
    const sourceUrl = `https://www.bilibili.com/video/${source.video_id}`;
    const result = await api.ingestUrl(listId, sourceUrl, true);
    trackJobs(
      result.queued.map((id) => ({
        id,
        producer: "ingest" as const,
        label: sourceUrl,
        contextKey: listId,
      })),
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 px-4 pt-4 pb-3">
        <form className="relative" onSubmit={handleSubmit}>
          <input
            type="text"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              setError(null);
            }}
            placeholder="Paste a Bilibili URL…"
            className="w-full rounded-full border border-border bg-white/80 py-2.5 pr-10 pl-4 text-sm text-ink placeholder:text-muted/50 outline-none focus:border-blue/40 focus:bg-white transition"
          />
          <button
            type="submit"
            disabled={!url.trim()}
            aria-label="Add source"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 flex h-7 w-7 items-center justify-center rounded-full text-muted transition disabled:opacity-0 enabled:hover:bg-blue enabled:hover:text-white enabled:hover:shadow-sm"
          >
            <MdArrowForward size={15} />
          </button>
        </form>
        {error && <p className="mt-1.5 px-4 text-xs text-rose-700">{error}</p>}
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto px-4 pb-4">
        {ingestJobs
          .filter((j) => !j.isTerminal || j.job.status === "failed" || j.job.status === "needs_auth")
          .map((item) => (
            <IngestingSourceRow
              key={item.job.id}
              stage={item.job.status}
              title={item.label}
              onDismiss={() => void dismissJob(item.job.id)}
            />
          ))}
        {currentSources.map((source) => (
          <SourceRow
            key={source.video_id}
            source={source}
            onOpen={() => onOpenSource(source)}
            onDelete={() => handleDelete(source)}
            onRerun={() => handleRerun(source)}
          />
        ))}
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ListDetailPage() {
  const { listId = "" } = useParams();
  const [listName, setListName] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [detailSource, setDetailSource] = useState<Source | null>(null);
  const [note, setNote] = useState<NoteContent | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const { sourcesW, labW, chatW, onMouseDownLeft, onMouseDownRight } = usePanelResize(
    containerRef,
    sourcesCollapsed,
  );

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [lists, nextSources] = await Promise.all([api.listLists(), api.listSources(listId)]);
        if (cancelled) return;
        const current = lists.find((l) => l.id === listId);
        setListName(current?.name ?? "List workspace");
        setSources(nextSources);
        setLoadError(null);
      } catch (err) {
        if (!cancelled) {
          setLoadError(toErrorMessage(err));
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [listId]);

  function handleOpenSource(source: Source) {
    setDetailSource(source);
    setNote(null);
    setTranscript(null);
    setTranscriptError(null);
    // Load note and transcript in parallel; rewrite relative image URLs to absolute API paths
    void api.getNoteContent(source.video_id).then((note) => {
      const rewritten = note.markdown.replace(
        /!\[([^\]]*)\]\(([^)]+)\)/g,
        (_match: string, alt: string, src: string) => {
          if (src.startsWith("http://") || src.startsWith("https://")) return _match;
          // src is relative to the attachments dir, e.g. "attachments/foo.jpg" → strip prefix
          const file = src.replace(/^attachments\//, "");
          return `![${alt}](/api/notes/${source.video_id}/attachments/${file})`;
        },
      );
      setNote({ ...note, markdown: rewritten });
    }).catch(() => setNote({ video_id: source.video_id, title: source.title, markdown: "" }));
    void api.getNoteTranscript(source.video_id)
      .then((res) => { setTranscript(res.text); })
      .catch(() => { setTranscriptError("Failed to load transcript"); });
  }

  async function handleRenameCommit(newName: string) {
    try {
      const updated = await api.updateList(listId, { name: newName });
      setListName(updated.name);
    } catch {
      // On failure the portal reverts its own draft via the name prop
    }
  }

  const panelBase = "flex shrink-0 flex-col overflow-hidden rounded-3xl border border-border bg-white/76 shadow-lg";

  return (
    <>
      <NavbarTitle name={listName} onCommit={handleRenameCommit} />

      <div
        ref={containerRef}
        className="fixed inset-x-0 top-14 bottom-0 z-0 box-border flex overflow-hidden px-4 pb-4"
      >
        {/* ── Sources panel ── */}
        <div
          style={
            sourcesCollapsed
              ? { width: `${COLLAPSED_PANEL}px`, minWidth: `${COLLAPSED_PANEL}px` }
              : { width: `${sourcesW}px`, minWidth: `${MIN_PANEL}px` }
          }
          className={panelBase}
        >
          <div className="flex shrink-0 items-center border-b border-border px-4 py-4">
            {!sourcesCollapsed && (
              <h2 className="m-0 flex-1 font-serif text-lg text-ink">Sources</h2>
            )}
            <button
              type="button"
              onClick={() => setSourcesCollapsed((v) => !v)}
              aria-label={sourcesCollapsed ? "Expand sources" : "Collapse sources"}
              className={`flex h-7 w-7 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink ${sourcesCollapsed ? "mx-auto" : ""}`}
            >
              {sourcesCollapsed ? <MdChevronRight size={16} /> : <MdChevronLeft size={16} />}
            </button>
          </div>

          {!sourcesCollapsed && (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              {loadError ? (
                <p className="m-0 px-4 py-3 text-sm text-rose-900">{loadError}</p>
              ) : detailSource ? (
                <SourcesViewerMode
                  source={detailSource}
                  note={note}
                  transcript={transcript}
                  transcriptError={transcriptError}
                  onClose={() => setDetailSource(null)}
                />
              ) : (
                <SourcesListMode
                  listId={listId}
                  sources={sources}
                  onOpenSource={handleOpenSource}
                />
              )}
            </div>
          )}
        </div>

        <Resizer onMouseDown={onMouseDownLeft} />

        {/* ── Chat panel ── */}
        <div
          style={{ width: `${chatW}px`, minWidth: `${MIN_PANEL}px` }}
          className={panelBase}
        >
          <SkeletonPanel
            title="Chat"
            note="List-scoped chat arrives in v1. This panel stays intentionally quiet until then."
          />
        </div>

        <Resizer onMouseDown={onMouseDownRight} />

        {/* ── Lab panel ── */}
        <div
          style={{ width: `${labW}px`, minWidth: `${MIN_PANEL}px` }}
          className={panelBase}
        >
          <SkeletonPanel
            title="Lab"
            note="Synthesis tools and overview export arrive in v1."
          />
        </div>
      </div>
    </>
  );
}
