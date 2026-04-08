import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useNavTitleContext } from "@/components/layout/NavTitleContext";

export function NavbarTitle({
  name,
  onCommit,
}: {
  name: string;
  onCommit: (newName: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);
  const navEl = useNavTitleContext();
  const inputRef = useRef<HTMLInputElement>(null);

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
