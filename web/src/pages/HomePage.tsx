import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ListGrid } from "../components/lists/ListGrid";
import { api, toErrorMessage } from "../lib/api";
import type { LocusList } from "../lib/types";

export function HomePage() {
  const navigate = useNavigate();
  const [lists, setLists] = useState<LocusList[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadLists() {
      try {
        const nextLists = await api.listLists();
        if (!cancelled) {
          setLists(nextLists);
        }
      } catch (nextError) {
        if (!cancelled) {
          setError(toErrorMessage(nextError));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    void loadLists();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleCreate() {
    setBusy(true);
    setError(null);
    try {
      const created = await api.createList("Untitled list");
      setLists((current) => [created, ...current]);
    } catch (nextError) {
      setError(toErrorMessage(nextError));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(list: LocusList) {
    if (!window.confirm(`Delete list "${list.name}"?`)) {
      return;
    }
    setError(null);
    try {
      await api.deleteList(list.id);
      setLists((current) => current.filter((entry) => entry.id !== list.id));
    } catch (nextError) {
      setError(toErrorMessage(nextError));
    }
  }

  return (
    <div className="form-stack">
      <section className="home-hero">
        <p className="home-hero__eyebrow">Capture. Distill. Revisit.</p>
        <h1 className="page-heading">Turn long-form video into a living, searchable notebook.</h1>
        <p className="page-lede">
          Build private list-based workspaces for courses, playlists, and research threads.
        </p>
        {error ? <p className="status-message error">{error}</p> : null}
      </section>
      {loading ? (
        <section className="panel">
          <p>Loading lists...</p>
        </section>
      ) : (
        <ListGrid
          busy={busy}
          lists={lists}
          onCreate={handleCreate}
          onDelete={handleDelete}
          onOpen={(list) => navigate(`/lists/${list.id}`)}
        />
      )}
    </div>
  );
}
