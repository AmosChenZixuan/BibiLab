import { useEffect, useState } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { ListGrid } from "@/components/lists/ListGrid";
import { DeleteListModal } from "@/components/lists/DeleteListModal";
import { RenameListModal } from "@/components/lists/RenameListModal";
import { ThumbnailPickerModal } from "@/components/lists/ThumbnailPickerModal";
import { api, toErrorMessageWithT } from "@/lib/api";
import type { BibilabList } from "@/lib/types";

export function HomePage() {
  const { t } = useLanguage();
  const [lists, setLists] = useState<BibilabList[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Modal targets
  const [deleteTarget, setDeleteTarget] = useState<BibilabList | null>(null);
  const [renameTarget, setRenameTarget] = useState<BibilabList | null>(null);
  const [thumbnailTarget, setThumbnailTarget] = useState<BibilabList | null>(null);

  const dialogOpen = deleteTarget !== null || renameTarget !== null || thumbnailTarget !== null;

  function updateLocalList(updated: BibilabList) {
    setLists((current) => current.map((entry) => (entry.id === updated.id ? updated : entry)));
  }

  useEffect(() => {
    const controller = new AbortController();
    async function loadLists() {
      try {
        const nextLists = await api.listLists({ signal: controller.signal });
        setLists(nextLists ?? []);
      } catch (nextError) {
        if (nextError instanceof Error && nextError.name === "AbortError") return;
        setError(toErrorMessageWithT(nextError, t));
      } finally {
        setLoading(false);
      }
    }
    void loadLists();
    return () => controller.abort();
  }, [t]);

  async function handleCreate() {
    setBusy(true);
    setError(null);
    try {
      const created = await api.createList(t("home.untitledList"));
      if (!created) return;
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

  async function handleRenameCommit(newName: string) {
    if (!renameTarget) return;
    setError(null);
    try {
      const updated = await api.updateList(renameTarget.id, { name: newName });
      if (!updated) return;
      updateLocalList(updated);
      setRenameTarget(null);
    } catch (nextError) {
      setError(toErrorMessageWithT(nextError, t));
    }
  }

  async function handleThumbnailSelect(thumbnailSourceId: string | null) {
    if (!thumbnailTarget) return;
    setError(null);
    try {
      const updated = await api.updateList(thumbnailTarget.id, {
        thumbnail_source_id: thumbnailSourceId,
      });
      if (!updated) return;
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
            <p>{t("home.loadingLists")}</p>
          ) : (
            <ListGrid
              busy={busy}
              lists={lists}
              onChangeThumbnail={(list) => setThumbnailTarget(list)}
              onCreate={handleCreate}
              onDelete={(list) => setDeleteTarget(list)}
              onRename={(list) => setRenameTarget(list)}
            />
          )}
        </div>
      </div>

      <DeleteListModal
        list={deleteTarget}
        onClose={() => setDeleteTarget(null)}
        onConfirm={handleDelete}
        open={deleteTarget !== null}
      />

      <RenameListModal
        list={renameTarget}
        onClose={() => setRenameTarget(null)}
        onCommit={handleRenameCommit}
        open={renameTarget !== null}
      />

      <ThumbnailPickerModal
        list={thumbnailTarget}
        onClose={() => setThumbnailTarget(null)}
        onSelect={handleThumbnailSelect}
        open={thumbnailTarget !== null}
      />
    </div>
  );
}
