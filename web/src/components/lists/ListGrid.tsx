import type { LocusList } from "../../lib/types";
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
      className="grid grid-cols-[repeat(auto-fill,248px)] justify-start gap-5 max-[820px]:grid-cols-1"
      aria-label="List grid"
    >
      <article className="h-[200px] w-[248px] rounded-[14px] border border-dashed border-[#c8c4bb] bg-white transition hover:border-pink hover:bg-[#fcf0f5] hover:text-[#c2607d] hover:shadow-card max-[820px]:w-full">
        <button
          aria-label="New list"
          className="flex h-full w-full flex-col items-center justify-center gap-2.5 border-0 bg-transparent text-[#6b6860]"
          disabled={busy}
          onClick={() => void onCreate()}
          type="button"
        >
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-current text-[22px] leading-none">
            +
          </span>
          <span className="text-[13.5px] font-medium tracking-[-0.01em]">New list</span>
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
