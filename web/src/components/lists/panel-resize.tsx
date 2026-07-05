import { useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";

export const MIN_PANEL = 280;
// Drag floor for the chat panel. Decoupled from the content cap: content self-caps at
// max-w-3xl (768) and centers, so the panel may go narrower and wrap hard without overflow.
export const MIN_CHAT_PANEL = 400;
// Initial on-load chat width — seeds the centered 768 reading column on first paint.
// The user can drag chat down to MIN_CHAT_PANEL to expand Sources/Lab past their default.
export const DEFAULT_CHAT = 768;
export const COLLAPSED_PANEL = 48;
const RESIZER_SIZE = 16;

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
  labCollapsed: boolean,
) {
  // ── Ratio refs (not pixel state) ─────────────────────────────────────────
  const sourcesRatio = useRef<number>(0);
  const labRatio = useRef<number>(0);
  // One-shot gate: rebalanceRatioRefs runs exactly once per hook instance, on the first
  // ResizeObserver measurement. Without this gate, every window resize would snap sources/
  // lab back to 50/50 and clobber the user's drag. Subsequent drags and resizes preserve
  // whatever the last set value was; the drag handler's clamp keeps it inside constraints.
  const ratiosRebalanced = useRef<boolean>(false);

  // Split the post-DEFAULT_CHAT remainder 50/50 — seeds chat at its comfortable reading
  // width while letting the user redistribute mass via the drag clamp (down to the lower
  // MIN_CHAT_PANEL floor). Runs from the first measurement only; the render's pixel-width
  // clamp below handles the infeasible narrow-viewport case.
  const rebalanceRatioRefs = useCallback((workspaceWidth: number) => {
    if (ratiosRebalanced.current) return;
    if (workspaceWidth <= 0 || sourcesCollapsed || labCollapsed) return;
    const availableRatio = 1 - DEFAULT_CHAT / workspaceWidth;
    sourcesRatio.current = availableRatio / 2;
    labRatio.current = availableRatio / 2;
    ratiosRebalanced.current = true;
  }, [sourcesCollapsed, labCollapsed]);

  const active = useRef<ActiveResizer>(null);
  const startX = useRef<number>(0);
  const startSourcesRatio = useRef<number>(0);
  const startLabRatio = useRef<number>(0);

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
  // useLayoutEffect (not useEffect) so the first paint already has the real container
  // width and the ratio refs. Otherwise the initial render produces sourcesWidth=0,
  // labWidth=0, chatWidth=0 — a one-frame flash of empty panel content.
  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Read padding once — it never changes for the lifetime of this container
    const styles = window.getComputedStyle(container);
    const paddingLeft = Number.parseFloat(styles.paddingLeft) || 0;
    const paddingRight = Number.parseFloat(styles.paddingRight) || 0;

    function measure() {
      if (!container) return;
      const contentWidth = Math.max(0, container.clientWidth - paddingLeft - paddingRight);
      const workspaceWidth = getResizableWorkspaceWidth(contentWidth);
      rebalanceRatioRefs(workspaceWidth);
      setContainerContentWidth(contentWidth);
    }

    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(container);

    return () => {
      observer.disconnect();
    };
  }, [containerRef, rebalanceRatioRefs]);

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
        const maxRatio = 1 - labRatio.current - MIN_CHAT_PANEL / workspaceWidth;
        sourcesRatio.current = clamp(
          startSourcesRatio.current + deltaRatio,
          minRatio,
          maxRatio,
        );
      } else {
        const minRatio = labCollapsed ? COLLAPSED_PANEL / workspaceWidth : MIN_PANEL / workspaceWidth;
        const maxRatio = 1 - sourcesRatio.current - MIN_CHAT_PANEL / workspaceWidth;
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
  }, [sourcesCollapsed, labCollapsed]);

  // ── Derived pixel widths (computed every render, not stored) ──────────────
  const workspaceWidth = getResizableWorkspaceWidth(containerContentWidth);
  // Clamp sources/lab to MIN_PANEL when expanded. The rebalance formula can yield
  // ratios that produce sub-MIN_PANEL widths (small viewports where availableRatio
  // is small or negative), and a user's drag can leave ratios in a state that
  // doesn't fit the current workspace on resize. Clamping here keeps the layout
  // valid; chat takes the residual (and may be < MIN_CHAT_PANEL when infeasible).
  const sourcesMin = sourcesCollapsed ? COLLAPSED_PANEL : MIN_PANEL;
  const labMin = labCollapsed ? COLLAPSED_PANEL : MIN_PANEL;
  const sourcesWidth = sourcesCollapsed
    ? COLLAPSED_PANEL
    : Math.max(sourcesMin, Math.round(sourcesRatio.current * workspaceWidth));
  const labWidth = labCollapsed
    ? COLLAPSED_PANEL
    : Math.max(labMin, Math.round(labRatio.current * workspaceWidth));
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
