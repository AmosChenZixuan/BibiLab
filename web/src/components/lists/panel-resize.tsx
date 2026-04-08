import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

export const MIN_PANEL = 280;
export const COLLAPSED_PANEL = 48;
export const RESIZER_SIZE = 16;

type ActiveResizer = "left" | "right" | null;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function getResizableWorkspaceWidth(containerWidth: number) {
  return Math.max(0, containerWidth - RESIZER_SIZE * 2);
}

export function usePanelResize(
  containerRef: RefObject<HTMLDivElement | null>,
  sourcesCollapsed: boolean,
) {
  // ── Ratio refs (not pixel state) ─────────────────────────────────────────
  const sourcesRatio = useRef<number>(1 / 3);
  const labRatio = useRef<number>(1 / 3);

  const active = useRef<ActiveResizer>(null);
  const startX = useRef<number>(0);
  const startSourcesRatio = useRef<number>(1 / 3);
  const startLabRatio = useRef<number>(1 / 3);

  // Container width — state so derived widths recompute reactively
  const [containerContentWidth, setContainerContentWidth] = useState<number>(0);
  // Ref to access current containerContentWidth inside non-react callbacks (mousemove)
  const containerContentWidthRef = useRef<number>(0);
  containerContentWidthRef.current = containerContentWidth;

  // Incremented during drag to force re-renders (refs alone don't trigger React re-renders)
  const [, forceRender] = useState<number>(0);

  const onMouseDownLeft = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    active.current = "left";
    startX.current = e.clientX;
    startSourcesRatio.current = sourcesRatio.current;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const onMouseDownRight = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    active.current = "right";
    startX.current = e.clientX;
    startLabRatio.current = labRatio.current;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  // ── Measure container size via ResizeObserver ─────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Read padding once — it never changes for the lifetime of this container
    const styles = window.getComputedStyle(container);
    const paddingLeft = Number.parseFloat(styles.paddingLeft) || 0;
    const paddingRight = Number.parseFloat(styles.paddingRight) || 0;

    function measure() {
      if (!container) return;
      setContainerContentWidth(
        Math.max(0, container.clientWidth - paddingLeft - paddingRight),
      );
    }

    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(container);

    return () => {
      observer.disconnect();
    };
  }, [containerRef]);

  // ── Drag handlers — update ratio refs ────────────────────────────────────
  useEffect(() => {
    // Persists across effect re-runs (survives containerContentWidth changes mid-drag)
    let lastRender = 0;

    function onMove(e: MouseEvent) {
      if (!active.current) return;
      const delta = e.clientX - startX.current;
      const workspaceWidth = getResizableWorkspaceWidth(containerContentWidthRef.current);
      if (!workspaceWidth) return;
      const deltaRatio = delta / workspaceWidth;

      if (active.current === "left") {
        const minRatio = sourcesCollapsed ? COLLAPSED_PANEL / workspaceWidth : MIN_PANEL / workspaceWidth;
        const maxRatio = 1 - labRatio.current - MIN_PANEL / workspaceWidth;
        sourcesRatio.current = clamp(
          startSourcesRatio.current + deltaRatio,
          minRatio,
          maxRatio,
        );
      } else {
        const minRatio = MIN_PANEL / workspaceWidth;
        const maxRatio = 1 - sourcesRatio.current - MIN_PANEL / workspaceWidth;
        labRatio.current = clamp(
          startLabRatio.current - deltaRatio,
          minRatio,
          maxRatio,
        );
      }

      // Throttle re-renders to ~60fps via requestAnimationFrame
      requestAnimationFrame(() => {
        if (Date.now() - lastRender >= 16) {
          lastRender = Date.now();
          forceRender((n) => n + 1);
        }
      });
    }

    function onUp() {
      if (!active.current) return;
      active.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      forceRender((n) => n + 1);
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [sourcesCollapsed]);

  // ── Derived pixel widths (computed every render, not stored) ──────────────
  const workspaceWidth = getResizableWorkspaceWidth(containerContentWidth);
  const sourcesWidth = sourcesCollapsed
    ? COLLAPSED_PANEL
    : Math.round(sourcesRatio.current * workspaceWidth);
  const labWidth = Math.round(labRatio.current * workspaceWidth);
  const chatWidth = workspaceWidth - sourcesWidth - labWidth;

  // Guard against unmount mid-drag
  useEffect(() => {
    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, []);

  return {
    sourcesW: sourcesWidth,
    labW: labWidth,
    chatW: chatWidth,
    onMouseDownLeft,
    onMouseDownRight,
  };
}

export function Resizer({ onMouseDown }: { onMouseDown: (e: React.MouseEvent) => void }) {
  return (
    <div
      className="shrink-0 cursor-col-resize self-stretch"
      style={{ width: `${RESIZER_SIZE}px` }}
      onMouseDown={onMouseDown}
    />
  );
}
