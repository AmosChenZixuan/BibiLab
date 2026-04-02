import type { LocusList } from "../../lib/types";
import { dangerButtonClass, eyebrowClass, mutedTextClass, sectionTitleClass } from "../../lib/ui";

type Props = {
  lists: LocusList[];
  onDelete: (list: LocusList) => Promise<void>;
  onOpen: (list: LocusList) => void;
  onCreate: () => Promise<void>;
  busy: boolean;
};

export function ListGrid({ lists, onDelete, onOpen, onCreate, busy }: Props) {
  return (
    <section
      className="grid grid-cols-[repeat(auto-fit,272px)] justify-start gap-4 max-[820px]:grid-cols-1"
      aria-label="List grid"
    >
      <article className="grid min-h-[220px] w-[272px] overflow-hidden rounded-card bg-[linear-gradient(160deg,rgba(245,140,185,0.72)_0%,rgba(243,162,198,0.68)_52%,rgba(150,227,255,0.7)_100%)] /* pink→sky list card gradient */ shadow-card max-[820px]:w-full">
        <button
          aria-label="Create new list"
          className="grid min-h-[220px] w-full content-between justify-items-start gap-4 border-0 bg-transparent p-[22px] text-left text-white"
          disabled={busy}
          onClick={() => void onCreate()}
          type="button"
        >
          <span className="text-[2.5rem] leading-none">+</span>
          <span className="font-serif text-2xl">Create new list</span>
        </button>
      </article>
      {lists.map((list) => (
        <article
          className="grid min-h-[220px] w-[272px] gap-3.5 rounded-card border border-border bg-surface p-5 shadow-card max-[820px]:w-full"
          key={list.id}
        >
          <button
            aria-label={`Open ${list.name}`}
            className="flex items-center justify-between border-0 bg-transparent p-0 text-left text-inherit"
            onClick={() => onOpen(list)}
            type="button"
          >
            <span className={eyebrowClass}>Notebook</span>
            <span className="rounded-full bg-sky/10 px-3 py-2 text-sm text-[#4e6f99]">Open</span>
          </button>
          <div>
            <h2 className={sectionTitleClass}>{list.name}</h2>
            <p className={mutedTextClass}>Created {new Date(list.created_at).toLocaleString()}</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              aria-label={`Delete ${list.name}`}
              className={dangerButtonClass}
              onClick={() => onDelete(list)}
              type="button"
            >
              Delete
            </button>
          </div>
        </article>
      ))}
    </section>
  );
}
