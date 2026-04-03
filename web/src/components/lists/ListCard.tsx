import { FiEdit2, FiImage, FiMoreVertical, FiTrash2 } from "react-icons/fi";
import { useNavigate } from "react-router-dom";

import type { LocusList } from "../../lib/types";
import { ContextMenu } from "../ui";

type Props = {
  list: LocusList;
  onRename?: (list: LocusList) => void;
  onChangeThumbnail?: (list: LocusList) => void;
  onDelete: (list: LocusList) => Promise<void> | void;
};

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
    <article className="group relative h-[200px] w-[248px] overflow-hidden rounded-[14px] shadow-card transition hover:-translate-y-0.5 hover:shadow-elevated max-[820px]:w-full">
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[linear-gradient(135deg,#f8b4c8_0%,#b3d9f0_100%)]"
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
        className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(0,0,0,0)_25%,rgba(0,0,0,0.62)_100%)]"
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[repeating-linear-gradient(0deg,rgba(255,255,255,0.06)_0px,rgba(255,255,255,0.06)_1px,transparent_1px,transparent_32px),repeating-linear-gradient(90deg,rgba(255,255,255,0.06)_0px,rgba(255,255,255,0.06)_1px,transparent_1px,transparent_32px)]"
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
        <div className="absolute right-0 bottom-0 left-0 z-[2] grid gap-[5px] p-3">
          <h2 className="m-0 line-clamp-2 text-[13.5px] font-semibold tracking-[-0.01em] text-white [text-shadow:0_1px_4px_rgba(0,0,0,0.25)]">
            {list.name}
          </h2>
          <div className="flex items-center gap-2 text-[11px] text-white/75">
            <span>{formatUpdatedDate(list.updated_at)}</span>
            <span className="h-[2.5px] w-[2.5px] rounded-full bg-white/40" />
            <span>{formatSourceCount(list.source_count)}</span>
          </div>
        </div>
      </button>
    </article>
  );
}
