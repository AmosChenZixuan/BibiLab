import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronDown, ChevronLeft, ChevronRight, Copy, Maximize2, Minus, Plus, RotateCcw } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import type { Artifact } from "@/lib/types";

interface ArtifactViewerProps {
  artifact: Artifact;
}

interface MindNode {
  label: string;
  children?: MindNode[];
}

interface MindTree {
  name?: string;
  root: MindNode;
}

const MIND_JSON_RE = /^```json\s*\n([\s\S]*?)\n```\s*$/m;
const MIND_MERMAID_RE = /^```mermaid\s*\n([\s\S]*?)\n```\s*$/m;

function parseMindTree(content: string): MindTree | null {
  // Try the new JSON format first.
  const j = MIND_JSON_RE.exec(content);
  if (j) {
    try {
      const parsed = JSON.parse(j[1]);
      if (parsed && typeof parsed === "object" && parsed.root && typeof parsed.root === "object") {
        return parsed as MindTree;
      }
    } catch {
      // fall through to Mermaid
    }
  }
  // Fallback: old Mermaid `flowchart TD` artifacts from the previous
  // mind-map implementation. The grammar is small: `id[label]` defines
  // a node, `A --> B[label]` defines an edge (parent → child).
  const m = MIND_MERMAID_RE.exec(content);
  if (m) return parseMermaidFlowchart(m[1]);
  return null;
}

function collectPaths(node: MindNode, path: string, depth: number, acc: string[] = []): string[] {
  if (depth >= 1) acc.push(path);
  (node.children ?? []).forEach((child, i) => {
    collectPaths(child, `${path}.${i}`, depth + 1, acc);
  });
  return acc;
}

function parseMermaidFlowchart(source: string): MindTree | null {
  const lines = source.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return null;
  if (!/^flowchart\s+TD$/i.test(lines[0])) return null;

  type Node = { id: string; label: string; children: string[] };
  const nodes = new Map<string, Node>();
  const edges: Array<[string, string]> = [];

  for (const line of lines.slice(1)) {
    // Edge: A --> B[Label]  or  A --> B
    const edge = /^(\w+)\s*-->\s*(\w+)(?:\[([^\]]+)\])?$/.exec(line);
    if (edge) {
      const [, src, dst, label] = edge;
      if (!nodes.has(src)) nodes.set(src, { id: src, label: src, children: [] });
      if (!nodes.has(dst)) {
        nodes.set(dst, { id: dst, label: label ?? dst, children: [] });
      }
      edges.push([src, dst]);
      continue;
    }
    // Standalone node: id[Label]
    const node = /^(\w+)\[([^\]]+)\]$/.exec(line);
    if (node) {
      const [, id, label] = node;
      const existing = nodes.get(id);
      if (existing) existing.label = label;
      else nodes.set(id, { id, label, children: [] });
    }
  }

  if (nodes.size === 0) return null;

  // Find the root: a node that's never a target of any edge.
  const targets = new Set(edges.map(([, dst]) => dst));
  const rootEntry = [...nodes.values()].find((n) => !targets.has(n.id)) ?? [...nodes.values()][0];

  const build = (node: Node): MindNode => ({
    label: node.label,
    children: edges
      .filter(([src]) => src === node.id)
      .map(([, dst]) => build(nodes.get(dst)!))
      .filter(Boolean),
  });

  return { name: undefined, root: build(rootEntry) };
}

const MindMapBlock: React.FC<{ content: string }> = ({ content }) => {
  const tree = useMemo(() => parseMindTree(content), [content]);
  // Default: only the root's direct children are expanded; everything
  // deeper starts collapsed so the initial view isn't an explosion.
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    if (!tree) return new Set();
    return new Set(collectPaths(tree.root, "0", 0));
  });
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [userInteracted, setUserInteracted] = useState(false);
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
    if (tree) walk(tree.root, "0");
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
      const safeScale = scale === 0 ? 1 : scale;
      const out: { key: string; d: string }[] = [];
      for (const { parent, child } of edges) {
        const p = nodeRefs.current.get(parent);
        const c = nodeRefs.current.get(child);
        if (!p || !c) continue;
        const pr = p.getBoundingClientRect();
        const cr = c.getBoundingClientRect();
        // Viewport → transformed-div local: subtract the inner div's
        // viewport origin and undo the scale (transform-origin is top-left).
        const x1 = (pr.right - ib.left) / safeScale;
        const y1 = (pr.top + pr.height / 2 - ib.top) / safeScale;
        const x2 = (cr.left - ib.left) / safeScale;
        const y2 = (cr.top + cr.height / 2 - ib.top) / safeScale;
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
  }, [edges, collapsed, scale, tx, ty, content, canvasSize.w, canvasSize.h]);

  // Fit-to-screen on first mount: scale to fit the tree in the canvas
  // and center it. Skipped if the user has already dragged/zoomed.
  useLayoutEffect(() => {
    if (userInteracted) return;
    fitToScreen();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [content]);

  if (!tree) {
    return (
      <div className="rounded-lg border border-pink/30 bg-pink/5 p-3 text-xs text-pink" role="alert">
        Mind map data is malformed — could not find a ```json tree fence.
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

  function isCollapsed(path: string) {
    return collapsed.has(path);
  }

  function onMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return;
    setUserInteracted(true);
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
    setUserInteracted(true);
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
    setUserInteracted(false);
  }

  function reset() {
    fitToScreen();
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
          className="origin-top-left"
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
          <div ref={treeRef} className="flex items-start gap-0 p-8">
            <TreeNode
              node={tree.root}
              path="0"
              depth={0}
              isCollapsed={isCollapsed}
              onToggle={toggle}
              nodeRefs={nodeRefs}
            />
          </div>
        </div>
      </div>

      {/* Control bar — top right, bigger icons so the +/- read clearly. */}
      <div className="absolute right-3 top-3 flex flex-col gap-1 rounded-full border border-border bg-white p-1 shadow-sm">
        <button
          type="button"
          aria-label="Zoom in"
          title="Zoom in"
          onClick={() => setScale((s) => Math.min(3, s + 0.2))}
          className="flex h-9 w-9 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
        >
          <Plus size={18} strokeWidth={2.5} />
        </button>
        <button
          type="button"
          aria-label="Zoom out"
          title="Zoom out"
          onClick={() => setScale((s) => Math.max(0.3, s - 0.2))}
          className="flex h-9 w-9 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
        >
          <Minus size={18} strokeWidth={2.5} />
        </button>
        <button
          type="button"
          aria-label="Reset view"
          title="Reset view"
          onClick={reset}
          className="flex h-9 w-9 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink"
        >
          <RotateCcw size={16} />
        </button>
      </div>
    </div>
  );
};

// Horizontal mind-map node (NotebookLM-style): root on the left, branches
// stacking vertically to its right, each branch's own children further
// to the right. The node card is a click-target (placeholder for future
// use). Expand/collapse is a SEPARATE round button — the card itself
// never toggles. Connector curves are drawn in an SVG overlay.
const TreeNode: React.FC<{
  node: MindNode;
  path: string;
  depth: number;
  isCollapsed: (path: string) => boolean;
  onToggle: (path: string) => void;
  nodeRefs: React.MutableRefObject<Map<string, HTMLElement>>;
}> = ({ node, path, depth, isCollapsed, onToggle, nodeRefs }) => {
  const children = node.children ?? [];
  const hasChildren = children.length > 0;
  const collapsed = isCollapsed(path);
  const isRoot = depth === 0;
  const isLeaf = depth >= 2;
  const cardClass = isRoot
    ? "rounded-2xl border-2 border-blue/30 bg-blue/10 px-5 py-3 shadow-sm"
    : isLeaf
      ? "rounded-lg border border-border bg-sky/8 px-3 py-1.5 text-sm shadow-sm"
      : "rounded-xl border border-border bg-white px-4 py-2 text-sm shadow-sm transition hover:shadow";

  const labelClass = isRoot
    ? "text-base font-semibold text-blue"
    : isLeaf
      ? "text-sm text-ink"
      : "text-sm font-medium text-ink";

  return (
    <div className="flex flex-row items-center" data-path={path}>
      {/* Node card — a click-target div (role=button) reserved for
          future use; today it does nothing on click. */}
      <div
        ref={(el) => {
          if (el) nodeRefs.current.set(path, el);
          else nodeRefs.current.delete(path);
        }}
        role="button"
        tabIndex={0}
        data-tree-node={path}
        className={`${cardClass} cursor-pointer select-none`}
      >
        <span className={labelClass}>{node.label}</span>
      </div>

      {/* Separate expand/collapse button — a round button at the joint
          of the connector curve. Hidden for leaves. */}
      {hasChildren && (
        <button
          type="button"
          onClick={() => onToggle(path)}
          aria-label={collapsed ? `Expand ${node.label}` : `Collapse ${node.label}`}
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
          by the parent SVG; no per-row lines needed here. */}
      {hasChildren && !collapsed && (
        <div className="ml-6 flex flex-col gap-3">
          {children.map((child, i) => (
            <TreeNode
              key={i}
              node={child}
              path={`${path}.${i}`}
              depth={depth + 1}
              isCollapsed={isCollapsed}
              onToggle={onToggle}
              nodeRefs={nodeRefs}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export function ArtifactViewer({ artifact }: ArtifactViewerProps) {
  const { t } = useLanguage();
  const [content, setContent] = useState<string | null>(null);

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

  function handleCopy() {
    if (!content) return;
    void navigator.clipboard.writeText(content);
  }

  const sourceCount = artifact.source_ids.length;
  const isMindMap = artifact.type === "mind_map";

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate font-serif text-base font-bold text-ink">{artifact.name}</p>
          <p className="m-0 text-xs text-muted">
            {t(sourceCount === 1 ? "lab.artifactViewer.basedOnSingular" : "lab.artifactViewer.basedOnPlural", { count: sourceCount })}
          </p>
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
            <MindMapBlock content={content} />
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  pre({ children }: { children?: ReactNode }) {
                    return (
                      <pre className="overflow-x-auto rounded-lg bg-border/30 p-3 text-xs">
                        {children}
                      </pre>
                    );
                  },
                }}
              >
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
