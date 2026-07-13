import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Minus, Plus, RotateCcw } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";

interface MindNode {
  label: string;
  // Verbatim transcript quote grounding this node, captured at generation
  // time. Absent on legacy artifacts → threaded as "" (today's behavior).
  evidence?: string;
  children?: MindNode[];
}

// Inner callback the TreeNode fires when a card is clicked. The
// page-level handler (MindMapAskInChat, in lib/chat-utils) is the
// wrapper that adds the artifact's source_ids; this file only knows
// about the topic + parent-topic pair plus the node's evidence quote.
type MindMapAskHandler = (topic: string, parentTopic: string | null, evidence: string) => void;

const MIND_JSON_RE = /^```json\s*\n([\s\S]*?)\n```\s*$/m;

// Shared cursor + focus ring for clickable node cards. Per-tier hover
// color is appended inline since each tier has a different base bg.
const INTERACTIVE_SUFFIX =
  "cursor-pointer transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue focus-visible:outline-offset-2";

const CONTROL_BTN =
  "flex h-9 w-9 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink";

function parseMindTree(content: string): MindNode | null {
  const j = MIND_JSON_RE.exec(content);
  if (!j) return null;
  try {
    const parsed = JSON.parse(j[1]);
    if (parsed?.root) return parsed.root as MindNode;
  } catch (err) {
    console.warn("parseMindTree: malformed JSON inside fence", err);
  }
  return null;
}

export const MindMapBlock: React.FC<{ content: string; onAskInChat?: MindMapAskHandler }> = ({ content, onAskInChat }) => {
  const { t } = useLanguage();
  const tree = useMemo(() => parseMindTree(content), [content]);
  // Default: only the root's direct children are expanded; everything
  // deeper starts collapsed so the initial view isn't an explosion.
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    if (!tree) return new Set();
    const acc: string[] = [];
    const walk = (node: MindNode, path: string) => {
      (node.children ?? []).forEach((child, i) => {
        const childPath = `${path}.${i}`;
        acc.push(childPath);
        walk(child, childPath);
      });
    };
    walk(tree, "0");
    return new Set(acc);
  });
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const dragRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const treeRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef<Map<string, HTMLElement>>(new Map());
  const [paths, setPaths] = useState<{ key: string; d: string }[]>([]);
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });

  // Edges from tree (parent path → list of child paths). Recomputed when
  // the tree or collapse state changes. Used to draw SVG connectors.
  const edges = useMemo(() => {
    const out: { parent: string; child: string }[] = [];
    function walk(node: MindNode, path: string) {
      for (let i = 0; i < (node.children ?? []).length; i++) {
        const childPath = `${path}.${i}`;
        out.push({ parent: path, child: childPath });
        walk(node.children![i], childPath);
      }
    }
    if (tree) walk(tree, "0");
    return out;
  }, [tree]);

  // Measure nodes and compute SVG connector paths. Re-runs after every
  // relevant state change (collapse, pan, zoom, content). Converts
  // viewport rects to the transformed div's local space (where the SVG
  // lives) so the curves stay anchored to the nodes after pan/zoom.
  useLayoutEffect(() => {
    const measure = () => {
      const container = containerRef.current;
      const inner = innerRef.current;
      if (!container || !inner) return;
      const cb = container.getBoundingClientRect();
      if (cb.width !== canvasSize.w || cb.height !== canvasSize.h) {
        setCanvasSize({ w: cb.width, h: cb.height });
      }
      const ib = inner.getBoundingClientRect();
      const out: { key: string; d: string }[] = [];
      for (const { parent, child } of edges) {
        const p = nodeRefs.current.get(parent);
        const c = nodeRefs.current.get(child);
        if (!p || !c) continue;
        const pr = p.getBoundingClientRect();
        const cr = c.getBoundingClientRect();
        // Viewport → transformed-div local: subtract the inner div's
        // viewport origin and undo the scale (transform-origin is top-left).
        const x1 = (pr.right - ib.left) / scale;
        const y1 = (pr.top + pr.height / 2 - ib.top) / scale;
        const x2 = (cr.left - ib.left) / scale;
        const y2 = (cr.top + cr.height / 2 - ib.top) / scale;
        const midX = (x1 + x2) / 2;
        out.push({
          key: `${parent}->${child}`,
          d: `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`,
        });
      }
      setPaths(out);
    };
    measure();
    // Re-measure on the next frame in case fonts/layouts shift.
    const raf = requestAnimationFrame(measure);
    const ro = new ResizeObserver(measure);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [edges, collapsed, scale, tx, ty, canvasSize.w, canvasSize.h]);

  if (!tree) {
    return (
      <div className="rounded-lg border border-pink/30 bg-pink/5 p-3 text-xs text-pink" role="alert">
        Mind map data is malformed or could not be parsed.
      </div>
    );
  }

  function toggle(path: string) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  function onMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return;
    dragRef.current = { x: e.clientX, y: e.clientY, tx, ty };
  }

  function onMouseMove(e: React.MouseEvent) {
    const d = dragRef.current;
    if (!d) return;
    setTx(d.tx + (e.clientX - d.x));
    setTy(d.ty + (e.clientY - d.y));
  }

  function onMouseUp() {
    dragRef.current = null;
  }

  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    const delta = -e.deltaY * 0.001;
    setScale((s) => Math.max(0.3, Math.min(3, s + delta)));
  }

  // Compute a fit-to-screen transform for the current tree. Used on
  // initial mount AND on the "Reset view" button so the user always
  // returns to a nicely framed view (not to scale=1, origin=(0,0)).
  // Uses offsetWidth/offsetHeight (layout size, unaffected by the
  // current transform) instead of getBoundingClientRect — otherwise
  // after a zoom, the "natural" size we measure is the post-transform
  // size, producing a wrong scale and a teleport to the top-left.
  function fitToScreen() {
    const tree = treeRef.current;
    const canvas = containerRef.current;
    if (!tree || !canvas) return;
    const tw = tree.offsetWidth;
    const th = tree.offsetHeight;
    const { width: cw, height: ch } = canvas.getBoundingClientRect();
    if (tw === 0 || th === 0) return;
    const s = Math.min(cw / tw, ch / th) * 0.9;
    setScale(s);
    setTx((cw - tw * s) / 2);
    setTy((ch - th * s) / 2);
  }

  return (
    <div className="relative h-[60vh] min-h-[420px] w-full select-none overflow-hidden rounded-2xl border border-border bg-white/40">
      {/* Tree canvas — panned/zoomed via CSS transform. */}
      <div
        ref={containerRef}
        className="absolute inset-0 cursor-grab active:cursor-grabbing"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onWheel={onWheel}
        data-testid="mindmap-canvas"
      >
        <div
          ref={innerRef}
          className="origin-top-left w-max"
          style={{
            transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
          }}
        >
          {/* SVG overlay for the smooth bezier connector curves. Lives in
              the transformed coordinate space so curves pan/zoom with the
              tree. pointer-events: none so clicks fall through to nodes. */}
          <svg
            className="pointer-events-none absolute left-0 top-0 overflow-visible"
            width={canvasSize.w}
            height={canvasSize.h}
            aria-hidden
          >
            {paths.map((p) => (
              <path
                key={p.key}
                d={p.d}
                stroke="rgb(147 197 253 / 0.6)"
                strokeWidth={2}
                fill="none"
              />
            ))}
          </svg>
          <div ref={treeRef} className="flex items-start p-8">
            <TreeNode
              node={tree}
              path="0"
              depth={0}
              parentLabel={null}
              isCollapsed={(p) => collapsed.has(p)}
              onToggle={toggle}
              onAskInChat={onAskInChat}
              nodeRefs={nodeRefs}
            />
          </div>
        </div>
      </div>

      {/* Control bar — top right, bigger icons so the +/- read clearly. */}
      <div className="absolute right-3 top-3 flex flex-col gap-1 rounded-full border border-border bg-white p-1 shadow-sm">
        {[
          { label: t("lab.mindMapViewer.zoomIn"), onClick: () => setScale((s) => Math.min(3, s + 0.2)), icon: Plus, iconSize: 18, strokeWidth: 2.5 },
          { label: t("lab.mindMapViewer.zoomOut"), onClick: () => setScale((s) => Math.max(0.3, s - 0.2)), icon: Minus, iconSize: 18, strokeWidth: 2.5 },
          { label: t("lab.mindMapViewer.resetView"), onClick: fitToScreen, icon: RotateCcw, iconSize: 16 },
        ].map(({ label, onClick, icon: Icon, iconSize, strokeWidth }) => (
          <button
            key={label}
            type="button"
            aria-label={label}
            title={label}
            onClick={onClick}
            className={CONTROL_BTN}
          >
            <Icon size={iconSize} strokeWidth={strokeWidth} />
          </button>
        ))}
      </div>
    </div>
  );
};

// Horizontal mind-map node (NotebookLM-style): root on the left, branches
// stacking vertically to its right, each branch's own children further
// to the right. The node card is a click-target — when `onAskInChat` is
// wired, clicking fires `discuss {label}, in the larger context of {parent}`
// into the chat (root passes parent=null → "Discuss {label}"). Expand/collapse
// is a SEPARATE round button — the card itself never toggles. Connector
// curves are drawn in an SVG overlay.
const TreeNode: React.FC<{
  node: MindNode;
  path: string;
  depth: number;
  parentLabel: string | null;
  isCollapsed: (path: string) => boolean;
  onToggle: (path: string) => void;
  onAskInChat?: MindMapAskHandler;
  nodeRefs: React.MutableRefObject<Map<string, HTMLElement>>;
}> = ({ node, path, depth, parentLabel, isCollapsed, onToggle, onAskInChat, nodeRefs }) => {
  const { t } = useLanguage();
  const children = node.children ?? [];
  const hasChildren = children.length > 0;
  const collapsed = isCollapsed(path);
  const isRoot = depth === 0;
  const isLeaf = depth >= 2;
  const clickable = !!onAskInChat;
  // Three visual tiers: root, internal (depth=1), leaf. The class names
  // are inline (not in a Record lookup) because the taxonomy is purely
  // a visual interpolation keyed by depth, not a semantic role.
  // `shrink-0` + `whitespace-nowrap` keep CJK labels on a single line —
  // without them, flexbox shrinks cards down to one character wide
  // because CJK breaks at every character boundary.
  // Hover state only applies when the card is clickable; non-clickable
  // cards stay visually flat.
  const interactiveSuffix = clickable
    ? isRoot
      ? `${INTERACTIVE_SUFFIX} hover:bg-blue/20`
      : isLeaf
        ? `${INTERACTIVE_SUFFIX} hover:bg-sky/20`
        : `${INTERACTIVE_SUFFIX} hover:brightness-95`
    : "";
  const cardClass = isRoot
    ? `shrink-0 rounded-2xl border-2 border-blue/30 bg-blue/10 px-5 py-3 whitespace-nowrap shadow-sm ${interactiveSuffix}`
    : isLeaf
      ? `shrink-0 rounded-lg border border-border bg-sky/8 px-3 py-1.5 text-sm whitespace-nowrap shadow-sm ${interactiveSuffix}`
      : `shrink-0 rounded-xl border border-border bg-white px-4 py-2 text-sm whitespace-nowrap shadow-sm ${interactiveSuffix}`;
  const labelClass = isRoot
    ? "text-base font-semibold whitespace-nowrap text-blue"
    : isLeaf
      ? "text-sm whitespace-nowrap text-ink"
      : "text-sm font-medium whitespace-nowrap text-ink";

  // The card is a <button> only when it has an action; otherwise it stays
  // a <div> so screen readers don't announce a non-action.
  const cardProps = {
    ref: (el: HTMLElement | null) => {
      if (el) nodeRefs.current.set(path, el);
      else nodeRefs.current.delete(path);
    },
    "data-tree-node": path,
    className: `${cardClass} select-none`,
    title: node.label,
  };

  return (
    <div className="flex flex-row items-center">
      {clickable ? (
        <button
          type="button"
          {...cardProps}
          onClick={() => onAskInChat?.(node.label, parentLabel, node.evidence ?? "")}
        >
          <span className={labelClass}>{node.label}</span>
        </button>
      ) : (
        <div {...cardProps}>
          <span className={labelClass}>{node.label}</span>
        </div>
      )}

      {/* Separate expand/collapse button — a round button at the joint
          of the connector curve. Hidden for leaves. */}
      {hasChildren && (
        <button
          type="button"
          onClick={() => onToggle(path)}
          aria-label={collapsed ? t("lab.mindMapViewer.expandNode", { label: node.label }) : t("lab.mindMapViewer.collapseNode", { label: node.label })}
          aria-expanded={!collapsed}
          className={`relative z-10 ml-2 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-blue/30 bg-white text-blue shadow-sm transition hover:bg-blue hover:text-white ${
            collapsed ? "" : "bg-blue/10"
          }`}
          data-testid={`mindmap-toggle-${path}`}
        >
          {collapsed ? (
            <ChevronRight size={16} strokeWidth={2.5} />
          ) : (
            <ChevronLeft size={16} strokeWidth={2.5} />
          )}
        </button>
      )}

      {/* Children column — only when expanded. Connectors are drawn
          by the parent SVG; no per-row lines needed. */}
      {hasChildren && !collapsed && (
        <div className="ml-6 flex flex-col gap-3">
          {children.map((child, i) => (
            <TreeNode
              key={i}
              node={child}
              path={`${path}.${i}`}
              depth={depth + 1}
              parentLabel={node.label}
              isCollapsed={isCollapsed}
              onToggle={onToggle}
              onAskInChat={onAskInChat}
              nodeRefs={nodeRefs}
            />
          ))}
        </div>
      )}
    </div>
  );
};
