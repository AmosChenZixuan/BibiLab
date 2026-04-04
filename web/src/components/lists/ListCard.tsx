import { FiEdit2, FiImage, FiMoreVertical, FiTrash2 } from "react-icons/fi";
import { useNavigate } from "react-router-dom";

import type { BibilabList } from "@/lib/types";
import { ContextMenu } from "@/components/ui";

type Props = {
  list: BibilabList;
  onRename?: (list: BibilabList) => void;
  onChangeThumbnail?: (list: BibilabList) => void;
  onDelete: (list: BibilabList) => Promise<void> | void;
};

export const PASTEL_COLORS = [
  "bg-pink-100",   // #fce7f3
  "bg-sky-100",    // #e0f2fe
  "bg-green-100",  // #dcfce7
  "bg-amber-100",  // #fef3c7
  "bg-violet-100", // #ede9fe
  "bg-orange-100", // #ffedd5
  "bg-teal-100",   // #f0fdf4
  "bg-slate-100",  // #f1f5f9
] as const;

export function nameToPastelIndex(name: string): number {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash) % PASTEL_COLORS.length;
}

function formatUpdatedDate(updatedAt: string) {
  return new Date(updatedAt).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function formatSourceCount(sourceCount: number) {
  return `${sourceCount} source${sourceCount === 1 ? "" : "s"}`;
}

export function ListCard({ list, onRename, onChangeThumbnail, onDelete }: Props) {
  const navigate = useNavigate();
  const menuItems = [
    ...(onRename
      ? [{ label: "Rename", icon: <FiEdit2 />, onClick: () => onRename(list) }]
      : []),
    ...(onChangeThumbnail
      ? [{ label: "Change thumbnail", icon: <FiImage />, onClick: () => onChangeThumbnail(list) }]
      : []),
    { label: "Delete list", icon: <FiTrash2 />, onClick: () => onDelete(list), variant: "danger" as const },
  ];

  return (
    <article className="group relative h-52 w-64 overflow-hidden rounded-2xl shadow-lg transition hover:-translate-y-0.5 hover:shadow-lg">
      <div
        aria-hidden="true"
        className={`absolute inset-0 ${!list.thumbnail_url ? PASTEL_COLORS[nameToPastelIndex(list.name)] : ""}`}
        style={
          list.thumbnail_url
            ? {
                backgroundImage: `linear-gradient(to bottom, rgba(17,17,17,0.02), rgba(17,17,17,0.02)), url("${list.thumbnail_url}")`,
                backgroundPosition: "center",
                backgroundSize: "cover",
              }
            : undefined
        }
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-linear-to-b from-transparent via-transparent to-black/60"
      />
      <div
        aria-hidden="true"
        className="absolute inset-0"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, rgba(255,255,255,0.06) 0px, rgba(255,255,255,0.06) 1px, transparent 1px, transparent 32px), repeating-linear-gradient(90deg, rgba(255,255,255,0.06) 0px, rgba(255,255,255,0.06) 1px, transparent 1px, transparent 32px)",
        }}
      />

      <ContextMenu
        items={menuItems}
        trigger={({ open, toggle, triggerRef }) => (
          <button
            aria-expanded={open}
            aria-label={`List actions for ${list.name}`}
            className={`absolute top-2 right-2 z-10 inline-flex h-7 w-7 items-center justify-center rounded-full border-0 text-white transition ${
              open ? "bg-white/90 text-ink" : "bg-white/20 opacity-100 md:opacity-0 md:group-hover:opacity-100"
            }`}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              toggle();
            }}
            ref={triggerRef}
            type="button"
          >
            <FiMoreVertical />
          </button>
        )}
      />

      <button
        aria-label={`Open ${list.name}`}
        className="absolute inset-0 border-0 bg-transparent p-0 text-left"
        onClick={() => navigate(`/lists/${list.id}`)}
        type="button"
      >
        <div className="absolute inset-x-0 bottom-0 z-10 grid gap-1 p-3">
          <h2 className="m-0 line-clamp-2 text-sm font-semibold tracking-tight text-white drop-shadow-sm">
            {list.name}
          </h2>
          <div className="flex items-center gap-2 text-xs text-white/75">
            <span>{formatUpdatedDate(list.updated_at)}</span>
            <span className="size-1 rounded-full bg-white/40" />
            <span>{formatSourceCount(list.source_count)}</span>
          </div>
        </div>
      </button>
    </article>
  );
}
