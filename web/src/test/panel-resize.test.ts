import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { usePanelResize, COLLAPSED_PANEL } from "@/components/lists/panel-resize";

// ─── ResizeObserver mock ───────────────────────────────────────────────────

const callbackMap = new Map<Element, ResizeObserverCallback>();
let originalResizeObserver: typeof ResizeObserver;

beforeEach(() => {
  callbackMap.clear();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  originalResizeObserver = global.ResizeObserver as any;
  const MockRO = function (callback: ResizeObserverCallback) {
    return {
      observe: (el: Element) => {
        callbackMap.set(el, callback);
        // jsdom does not auto-fire ResizeObserver callbacks —
        // tests call triggerResize(el, width) to simulate size changes.
      },
      disconnect: vi.fn(),
      unobserve: vi.fn(),
    };
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  global.ResizeObserver = MockRO as any;
});

afterEach(() => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  global.ResizeObserver = originalResizeObserver as any;
});

function makeDiv(width = 1440) {
  const el = document.createElement("div");
  Object.defineProperty(el, "clientWidth", { value: width, configurable: true });
  return el;
}

function triggerResize(el: Element, width: number) {
  const cb = callbackMap.get(el);
  if (!cb) return;
  act(() => {
    Object.defineProperty(el, "clientWidth", { value: width, configurable: true });
    cb(
      [{ contentRect: { width }, target: el }] as ResizeObserverEntry[],
      vi.fn() as unknown as ResizeObserver,
    );
  });
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("usePanelResize", () => {
  // ── AC1: sourcesW is derived from ratio, not stored as state ───────────────

  test("sourcesW derives from ratio — initial at 1440px", () => {
    const div = makeDiv(1440);
    const { result } = renderHook(() =>
      usePanelResize({ current: div } as React.RefObject<HTMLDivElement>, false, false),
    );

    triggerResize(div, 1440);

    // workspaceWidth = 1440 - 32 = 1408; ratio = 1/3 → sourcesW ≈ 469
    expect(result.current.sourcesW).toBeGreaterThan(400);
    expect(result.current.sourcesW).toBeLessThan(520);
    expect(result.current.labW).toBeGreaterThan(400);
    expect(result.current.labW).toBeLessThan(520);
  });

  // ── AC3 & AC4: Ratio preserved on container resize — proportional scaling ───

  test("shrinking container scales sourcesW proportionally — ratio preserved", () => {
    const div = makeDiv(1440);
    const { result } = renderHook(() =>
      usePanelResize({ current: div } as React.RefObject<HTMLDivElement>, false, false),
    );

    triggerResize(div, 1440);

    // At 1440: workspaceWidth=1408, sourcesW≈469, ratio=1/3
    const sourcesW_1440 = result.current.sourcesW;
    expect(sourcesW_1440).toBeGreaterThan(400);
    expect(sourcesW_1440).toBeLessThan(520);

    // Shrink to 800: workspaceWidth=768
    triggerResize(div, 800);

    // Ratio preserved (1/3) → sourcesW ≈ 768/3 = 256 (clamped to MIN by hook math)
    // If ratio were reset, sourcesW would be the same absolute pixel value (no change)
    // The key test: at 800px, sourcesW is smaller than at 1440px (proportional)
    expect(result.current.sourcesW).toBeLessThan(sourcesW_1440);

    // chatW + sourcesW + labW === 768
    const total = result.current.sourcesW + result.current.chatW + result.current.labW;
    expect(total).toBe(768);
  });

  test("returning to original size restores proportional widths", () => {
    const div = makeDiv(1440);
    const { result } = renderHook(() =>
      usePanelResize({ current: div } as React.RefObject<HTMLDivElement>, false, false),
    );

    triggerResize(div, 1440);
    const at1440 = result.current.sourcesW;

    triggerResize(div, 800);
    const at800 = result.current.sourcesW;
    expect(at800).toBeLessThan(at1440); // shrunk proportionally

    triggerResize(div, 1440);
    // At 1440 again, sourcesW should match the 1440 value (ratio preserved)
    expect(result.current.sourcesW).toBe(at1440);
  });

  // ── AC5: Panel widths sum to workspaceWidth ───────────────────────────────

  test("chatW + sourcesW + labW === workspaceWidth at 1440px", () => {
    const div = makeDiv(1440);
    const { result } = renderHook(() =>
      usePanelResize({ current: div } as React.RefObject<HTMLDivElement>, false, false),
    );

    triggerResize(div, 1440);

    const workspaceWidth = 1408;
    const total = result.current.sourcesW + result.current.chatW + result.current.labW;
    expect(total).toBe(workspaceWidth);
  });

  test("sum still holds after container shrink to 800px", () => {
    const div = makeDiv(800);
    const { result } = renderHook(() =>
      usePanelResize({ current: div } as React.RefObject<HTMLDivElement>, false, false),
    );

    triggerResize(div, 800);

    const workspaceWidth = 768;
    const total = result.current.sourcesW + result.current.chatW + result.current.labW;
    expect(total).toBe(workspaceWidth);
  });

  test("sum still holds at very small viewport (600px)", () => {
    const div = makeDiv(600);
    const { result } = renderHook(() =>
      usePanelResize({ current: div } as React.RefObject<HTMLDivElement>, false, false),
    );

    triggerResize(div, 600);

    const workspaceWidth = 568;
    const total = result.current.sourcesW + result.current.chatW + result.current.labW;
    expect(total).toBe(workspaceWidth);
  });

  // ── AC6: Collapse behavior unaffected ────────────────────────────────────

  test("sourcesCollapsed=true renders COLLAPSED_PANEL (48px)", () => {
    const div = makeDiv(1440);
    const { result } = renderHook(() =>
      usePanelResize({ current: div } as React.RefObject<HTMLDivElement>, true, false),
    );

    triggerResize(div, 1440);

    expect(result.current.sourcesW).toBe(COLLAPSED_PANEL);
  });

  // ── AC7: Return types ─────────────────────────────────────────────────────

  test("sourcesW, labW, chatW are numbers", () => {
    const div = makeDiv(1440);
    const { result } = renderHook(() =>
      usePanelResize({ current: div } as React.RefObject<HTMLDivElement>, false, false),
    );
    triggerResize(div, 1440);
    expect(typeof result.current.sourcesW).toBe("number");
    expect(typeof result.current.labW).toBe("number");
    expect(typeof result.current.chatW).toBe("number");
  });
});
