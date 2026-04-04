import { useEffect, useState } from "react";

import { ListGrid } from "@/components/lists/ListGrid";
import { api, toErrorMessage } from "@/lib/api";
import type { LocusList, Source } from "@/lib/types";
import { Button, Input, Modal, Panel } from "@/components/ui";

export function HomePage() {
  const [lists, setLists] = useState<LocusList[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<LocusList | null>(null);
  const [renameTarget, setRenameTarget] = useState<LocusList | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [thumbnailTarget, setThumbnailTarget] = useState<LocusList | null>(null);
  const [thumbnailSources, setThumbnailSources] = useState<Source[]>([]);
  const [thumbnailLoading, setThumbnailLoading] = useState(false);
  const dialogOpen = deleteTarget !== null || renameTarget !== null || thumbnailTarget !== null;

  function updateLocalList(updated: LocusList) {
    setLists((current) => current.map((entry) => (entry.id === updated.id ? updated : entry)));
  }

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
    setError(null);
    try {
      await api.deleteList(list.id);
      setLists((current) => current.filter((entry) => entry.id !== list.id));
      setDeleteTarget(null);
    } catch (nextError) {
      setError(toErrorMessage(nextError));
    }
  }

  async function handleRenameCommit() {
    if (!renameTarget) {
      return;
    }

    const trimmed = renameDraft.trim();
    if (!trimmed) {
      setRenameDraft(renameTarget.name);
      return;
    }
    if (trimmed === renameTarget.name) {
      setRenameTarget(null);
      return;
    }

    setError(null);
    try {
      const updated = await api.updateList(renameTarget.id, { name: trimmed });
      updateLocalList(updated);
      setRenameTarget(null);
    } catch (nextError) {
      setError(toErrorMessage(nextError));
      setRenameDraft(renameTarget.name);
    }
  }

  async function openThumbnailDialog(list: LocusList) {
    setThumbnailTarget(list);
    setThumbnailSources([]);
    setThumbnailLoading(true);
    setError(null);
    try {
      const sources = await api.listSources(list.id);
      setThumbnailSources(sources);
    } catch (nextError) {
      setError(toErrorMessage(nextError));
    } finally {
      setThumbnailLoading(false);
    }
  }

  async function handleThumbnailSelect(thumbnailSourceId: string | null) {
    if (!thumbnailTarget) {
      return;
    }

    setError(null);
    try {
      const updated = await api.updateList(thumbnailTarget.id, {
        thumbnail_source_id: thumbnailSourceId,
      });
      updateLocalList(updated);
      setThumbnailTarget(null);
    } catch (nextError) {
      setError(toErrorMessage(nextError));
    }
  }

  return (
    <div className="grid gap-4">
      <div
        aria-hidden={dialogOpen}
        data-testid="home-page-content"
        className={dialogOpen ? "pointer-events-none select-none blur-sm transition" : "transition"}
      >
        <section className="grid gap-3">
          <span className="text-xs font-semibold uppercase tracking-widest text-muted">My Lists</span>
          {error ? <p className="m-0 text-sm text-rose-900">{error}</p> : null}
        </section>
        <div className="mt-4">
          {loading ? (
            <Panel variant="app">
              <p>Loading lists...</p>
            </Panel>
          ) : (
            <ListGrid
              busy={busy}
              lists={lists}
              onChangeThumbnail={openThumbnailDialog}
              onCreate={handleCreate}
            onDelete={(list) => {
              setDeleteTarget(list);
            }}
            onRename={(list) => {
              setRenameTarget(list);
              setRenameDraft(list.name);
              }}
            />
          )}
        </div>
      </div>

      <Modal
        footer={
          <>
            <Button onClick={() => setDeleteTarget(null)} size="sm" variant="ghost">
              Cancel
            </Button>
            <Button
              onClick={() => {
                if (deleteTarget) {
                  void handleDelete(deleteTarget);
                }
              }}
              size="sm"
              variant="danger"
            >
              Delete
            </Button>
          </>
        }
        onClose={() => setDeleteTarget(null)}
        open={deleteTarget !== null}
        size="lg"
        title="Delete list"
      >
        <div className="rounded-2xl border border-rose-300/50 bg-rose-50 p-4 text-sm text-rose-900">
          <p className="m-0 text-base font-semibold tracking-tight">This cannot be undone</p>
          <p className="mt-1.5 mb-0 leading-6">
            {deleteTarget
              ? `"${deleteTarget.name}" and its ${deleteTarget.source_count} sources will be permanently removed.`
              : ""}
          </p>
        </div>
      </Modal>

      <Modal
        footer={
          <>
            <Button
              onClick={() => {
                setRenameTarget(null);
                setRenameDraft("");
              }}
              size="sm"
              variant="ghost"
            >
              Cancel
            </Button>
            <Button onClick={() => void handleRenameCommit()} size="sm" variant="primary">
              Save
            </Button>
          </>
        }
        onClose={() => setRenameTarget(null)}
        open={renameTarget !== null}
        size="lg"
        title="Rename list"
      >
        <div className="relative h-100 overflow-hidden rounded-3xl bg-pink-100 shadow-lg">
          {renameTarget?.thumbnail_url ? (
            <div
              className="absolute inset-0 bg-cover bg-center"
              style={{ backgroundImage: `url("${renameTarget.thumbnail_url}")` }}
            />
          ) : null}
          <div className="absolute inset-0 bg-linear-to-t from-black/65 via-black/20 to-transparent" />
          <div className="absolute inset-x-6 bottom-6 z-10">
            <span className="block text-3xl font-semibold tracking-tighter leading-tight text-white">
              {renameDraft || renameTarget?.name || "Untitled list"}
            </span>
          </div>
        </div>
        <div className="">
          <label className="grid gap-2">
            <span className="text-xs font-semibold uppercase tracking-widest text-muted">List name</span>
            <Input
              aria-label="List name"
              autoFocus
              className="select-text rounded-2xl bg-white/92 px-4 py-3 text-2xl leading-tight font-normal tracking-normal text-ink focus:border-blue/25 focus:ring-2 focus:ring-sky/10"
              placeholder="Untitled list"
              onChange={(event) => setRenameDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  void handleRenameCommit();
                }
              }}
              value={renameDraft}
            />
          </label>
        </div>
      </Modal>

      <Modal
        onClose={() => setThumbnailTarget(null)}
        open={thumbnailTarget !== null}
        size="lg"
        title="Choose thumbnail"
      >
        {thumbnailLoading ? (
          <p className="m-0 text-sm text-muted">Loading sources...</p>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            <button
              className="aspect-video overflow-hidden rounded-2xl border border-border bg-black/30 p-0 text-left shadow-lg transition hover:-translate-y-0.5 hover:shadow-lg"
              onClick={() => void handleThumbnailSelect(null)}
              type="button"
            >
              <div className="flex h-full items-end bg-cover bg-center p-2">
                <span className="block truncate text-xs font-semibold text-white">No cover</span>
              </div>
            </button>
            {thumbnailSources.map((source) => (
              <button
                className="aspect-video overflow-hidden rounded-2xl border border-border bg-black/30 p-0 text-left shadow-lg transition hover:-translate-y-0.5 hover:shadow-lg"
                key={source.video_id}
                onClick={() => void handleThumbnailSelect(source.video_id)}
                type="button"
              >
                <div
                  className="flex h-full items-end bg-cover bg-center p-2"
                  style={{ backgroundImage: `linear-gradient(to top, rgba(0,0,0,0.5), transparent), url("/api/covers/${source.video_id}")` }}
                >
                  <span className="block truncate text-xs font-semibold text-white">{source.title}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </Modal>
    </div>
  );
}
