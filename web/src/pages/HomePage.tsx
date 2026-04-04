import { useEffect, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { ListGrid } from "@/components/lists/ListGrid";
import { api, toErrorMessageWithT } from "@/lib/api";
import type { BibilabList, Source } from "@/lib/types";
import { Button, Input, Modal, Panel } from "@/components/ui";

export function HomePage() {
  const { t } = useLanguage();
  const [lists, setLists] = useState<BibilabList[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<BibilabList | null>(null);
  const [renameTarget, setRenameTarget] = useState<BibilabList | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [thumbnailTarget, setThumbnailTarget] = useState<BibilabList | null>(null);
  const [thumbnailSources, setThumbnailSources] = useState<Source[]>([]);
  const [thumbnailLoading, setThumbnailLoading] = useState(false);
  const dialogOpen = deleteTarget !== null || renameTarget !== null || thumbnailTarget !== null;

  function updateLocalList(updated: BibilabList) {
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
          setError(toErrorMessageWithT(nextError, t));
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
      const created = await api.createList(t("home.untitledList"));
      setLists((current) => [created, ...current]);
    } catch (nextError) {
      setError(toErrorMessageWithT(nextError, t));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(list: BibilabList) {
    setError(null);
    try {
      await api.deleteList(list.id);
      setLists((current) => current.filter((entry) => entry.id !== list.id));
      setDeleteTarget(null);
    } catch (nextError) {
      setError(toErrorMessageWithT(nextError, t));
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
      setError(toErrorMessageWithT(nextError, t));
      setRenameDraft(renameTarget.name);
    }
  }

  async function openThumbnailDialog(list: BibilabList) {
    setThumbnailTarget(list);
    setThumbnailSources([]);
    setThumbnailLoading(true);
    setError(null);
    try {
      const sources = await api.listSources(list.id);
      setThumbnailSources(sources);
    } catch (nextError) {
      setError(toErrorMessageWithT(nextError, t));
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
      setError(toErrorMessageWithT(nextError, t));
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
          <span className="text-xs font-semibold uppercase tracking-widest text-muted">{t("home.myLists")}</span>
          {error ? <p className="m-0 text-sm text-rose-900">{error}</p> : null}
        </section>
        <div className="mt-4">
          {loading ? (
            <Panel variant="app">
              <p>{t("home.loadingLists")}</p>
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
              {t("common.cancel")}
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
              {t("common.delete")}
            </Button>
          </>
        }
        onClose={() => setDeleteTarget(null)}
        open={deleteTarget !== null}
        size="lg"
        title={t("home.deleteList")}
      >
        <div className="rounded-2xl border border-rose-300/50 bg-rose-50 p-4 text-sm text-rose-900">
          <p className="m-0 text-base font-semibold tracking-tight">{t("home.cannotUndo")}</p>
          <p className="mt-1.5 mb-0 leading-6">
            {deleteTarget
              ? t("home.deleteConfirm", { name: deleteTarget.name, count: deleteTarget.source_count })
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
              {t("common.cancel")}
            </Button>
            <Button onClick={() => void handleRenameCommit()} size="sm" variant="primary">
              {t("common.save")}
            </Button>
          </>
        }
        onClose={() => setRenameTarget(null)}
        open={renameTarget !== null}
        size="lg"
        title={t("home.renameList")}
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
              {renameDraft || renameTarget?.name || t("home.untitledList")}
            </span>
          </div>
        </div>
        <div className="">
          <label className="grid gap-2">
            <span className="text-xs font-semibold uppercase tracking-widest text-muted">{t("home.listName")}</span>
            <Input
              aria-label="List name"
              autoFocus
              className="select-text rounded-2xl bg-white/92 px-4 py-3 text-2xl leading-tight font-normal tracking-normal text-ink focus:border-blue/25 focus:ring-2 focus:ring-sky/10"
              placeholder={t("home.untitledList")}
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
        title={t("home.chooseThumbnail")}
      >
        {thumbnailLoading ? (
          <p className="m-0 text-sm text-muted">{t("home.loadingSources")}</p>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            <button
              className="aspect-video overflow-hidden rounded-2xl border border-border bg-black/30 p-0 text-left shadow-lg transition hover:-translate-y-0.5 hover:shadow-lg"
              onClick={() => void handleThumbnailSelect(null)}
              type="button"
            >
              <div className="flex h-full items-end bg-cover bg-center p-2">
                <span className="block truncate text-xs font-semibold text-white">{t("home.noCover")}</span>
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
