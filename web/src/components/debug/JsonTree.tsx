import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function JsonTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null || value === undefined) {
    return <span className="text-(--color-muted)">—</span>;
  }
  if (typeof value === "string") {
    return <ReactMarkdown remarkPlugins={[remarkGfm]}>{value}</ReactMarkdown>;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return <span>{String(value)}</span>;
  }
  if (Array.isArray(value)) {
    return (
      <ul className="list-disc pl-5 space-y-1">
        {value.map((v, i) => (
          <li key={i}>
            <JsonTree value={v} depth={depth + 1} />
          </li>
        ))}
      </ul>
    );
  }
  if (typeof value === "object") {
    return <ObjectView value={value as Record<string, unknown>} depth={depth} />;
  }
  return <span>{String(value)}</span>;
}

function ObjectView({ value, depth }: { value: Record<string, unknown>; depth: number }) {
  const [open, setOpen] = useState(depth < 1);
  const entries = Object.entries(value);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="text-xs text-(--color-muted) hover:text-(--color-ink) font-mono"
      >
        {open ? "▼" : "▶"} {"{"}
      </button>
      {open ? (
        <dl className="pl-4 mt-1 space-y-0.5">
          {entries.map(([k, v]) => (
            <div key={k} className="flex gap-2 text-sm">
              <dt className="text-(--color-muted) font-mono">{k}</dt>
              <dd className="flex-1">
                <JsonTree value={v} depth={depth + 1} />
              </dd>
            </div>
          ))}
        </dl>
      ) : (
        <CollapsedEntries entries={entries} />
      )}
    </div>
  );
}

function CollapsedEntries({ entries }: { entries: [string, unknown][] }) {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const expanded = expandedKey ? entries.find(([k]) => k === expandedKey) : null;
  return (
    <span className="font-mono text-xs">
      {entries.map(([k], i) => (
        <span key={k}>
          {i > 0 ? ", " : ""}
          <button
            type="button"
            onClick={() => setExpandedKey((cur) => (cur === k ? null : k))}
            className="text-(--color-muted) hover:text-(--color-ink) underline-offset-2 hover:underline"
          >
            {k}
          </button>
        </span>
      ))}
      {" }"}
      {expanded && (
        <div className="block pl-3 mt-0.5">
          <JsonTree value={expanded[1]} depth={0} />
        </div>
      )}
    </span>
  );
}
