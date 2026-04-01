import type { LocusList } from "../../lib/types";

type Props = {
  lists: LocusList[];
  onDelete: (list: LocusList) => Promise<void>;
  onOpen: (list: LocusList) => void;
  onCreate: () => Promise<void>;
  busy: boolean;
};

export function ListGrid({ lists, onDelete, onOpen, onCreate, busy }: Props) {
  return (
    <section className="card-grid" aria-label="List grid">
      <article className="list-card list-card--create">
        <button
          aria-label="Create new list"
          className="create-list-tile"
          disabled={busy}
          onClick={() => void onCreate()}
          type="button"
        >
          <span className="create-list-tile__plus">+</span>
          <span className="create-list-tile__title">Create new list</span>
        </button>
      </article>
      {lists.map((list) => (
        <article className="list-card" key={list.id}>
          <button className="list-card__open" onClick={() => onOpen(list)} type="button">
            <span className="list-card__eyebrow">Notebook</span>
            <span className="list-card__chevron">Open</span>
          </button>
          <div>
            <h2 className="list-card__title">{list.name}</h2>
            <p className="page-lede">Created {new Date(list.created_at).toLocaleString()}</p>
          </div>
          <div className="inline-actions">
            <button
              aria-label={`Delete ${list.name}`}
              className="danger-button"
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
