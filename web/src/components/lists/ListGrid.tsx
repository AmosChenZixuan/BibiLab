import { FiPlus } from "react-icons/fi";
import type { LocusList } from "@/lib/types";
import { ListCard } from "./ListCard";

type Props = {
  lists: LocusList[];
  onDelete: (list: LocusList) => Promise<void> | void;
  onRename?: (list: LocusList) => void;
  onChangeThumbnail?: (list: LocusList) => void;
  onCreate: () => Promise<void>;
  busy: boolean;
};

export function ListGrid({
  lists,
  onDelete,
  onRename,
  onChangeThumbnail,
  onCreate,
  busy,
}: Props) {
  return (
    <section
      className="grid justify-start gap-5 max-sm:justify-center"
      aria-label="List grid"
      style={{ gridTemplateColumns: "repeat(auto-fill, 16rem)" }}
    >
      <article className="h-52 w-64 rounded-2xl border border-dashed border-stone-300 bg-white transition hover:border-pink hover:bg-pink/8 hover:text-rose-500 hover:shadow-lg">
        <button
          aria-label="New list"
          className="flex h-full w-full flex-col items-center justify-center gap-2.5 border-0 bg-transparent text-stone-500"
          disabled={busy}
          onClick={() => void onCreate()}
          type="button"
        >
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-current">
            <FiPlus className="size-5" aria-hidden="true" />
          </span>
          <span className="text-sm font-medium tracking-tight">New list</span>
        </button>
      </article>
      {lists.map((list) => (
        <ListCard
          key={list.id}
          list={list}
          onChangeThumbnail={onChangeThumbnail}
          onDelete={onDelete}
          onRename={onRename}
        />
      ))}
    </section>
  );
}
