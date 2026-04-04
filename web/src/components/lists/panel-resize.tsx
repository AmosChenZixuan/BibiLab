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

export function usePanelResize(
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

  // Guard against unmount mid-drag (e.g. navigation away)
  useEffect(() => {
    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, []);

  return { sourcesW, labW, chatW, onMouseDownLeft, onMouseDownRight };
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
