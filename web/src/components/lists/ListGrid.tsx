import { FiPlus } from "react-icons/fi";

import { useLanguage } from "@/app/LanguageContext";
import type { BibilabList } from "@/lib/types";
import { ListCard } from "./ListCard";

type Props = {
  lists: BibilabList[];
  onDelete: (list: BibilabList) => Promise<void> | void;
  onRename?: (list: BibilabList) => void;
  onChangeThumbnail?: (list: BibilabList) => void;
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
  const { t } = useLanguage();
  return (
    <section
      className="grid justify-start gap-5 max-sm:justify-center"
      aria-label="My Lists"
      style={{ gridTemplateColumns: "repeat(auto-fill, 16rem)" }}
    >
      <article className="h-52 w-64 rounded-2xl border border-dashed border-border bg-white transition hover:border-pink hover:bg-pink/8 hover:text-pink hover:shadow-lg">
        <button
          aria-label="New list"
          className="flex h-full w-full flex-col items-center justify-center gap-2.5 border-0 bg-transparent text-muted"
          disabled={busy}
          onClick={() => void onCreate()}
          type="button"
        >
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-current">
            <FiPlus className="size-5" aria-hidden="true" />
          </span>
          <span className="text-sm font-medium tracking-tight">{t("home.createList")}</span>
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
