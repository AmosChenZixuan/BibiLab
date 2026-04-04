/**
 * ListDetailPage — redesign per docs/specs/2026-04-03-list-detail-redesign.md
 *
 * Slice 1: Page shell, resize, skeleton panels.
 * Chat and Lab panels render as skeletons. Sources panel shows its header and
 * collapse toggle with an empty body. No API wiring yet.
 */

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { useParams } from "react-router-dom";
import { MdChevronLeft, MdChevronRight } from "react-icons/md";

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

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ListDetailPage() {
  const { listId = "" } = useParams();
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const { sourcesW, labW, chatW, onMouseDownLeft, onMouseDownRight } = usePanelResize(
    containerRef,
    sourcesCollapsed,
  );

  // Placeholder: list name loaded in Slice 2, sources loaded in Slice 3
  void listId;

  const panelBase = "flex shrink-0 flex-col overflow-hidden rounded-3xl border border-border bg-white/76 shadow-lg";

  return (
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
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden" />
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
  );
}
