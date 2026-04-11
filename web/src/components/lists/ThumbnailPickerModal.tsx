import { useEffect, useRef, useState } from "react";

import type { BibilabList, Source } from "@/lib/types";
import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import { Modal } from "@/components/ui";

interface ThumbnailPickerModalProps {
  list: BibilabList | null;
  open: boolean;
  onClose: () => void;
  onSelect: (thumbnailSourceId: string | null) => Promise<void>;
}

export function ThumbnailPickerModal({ list, open, onClose, onSelect }: ThumbnailPickerModalProps) {
  const { t } = useLanguage();
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!open || !list) return;
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setSources([]);
    api.listSources(list.id, { signal: controller.signal }).then((next) => {
      setSources(next ?? []);
    }).catch(() => {
      setSources([]);
    }).finally(() => {
      setLoading(false);
    });
    return () => controller.abort();
  }, [open, list]);

  return (
    <Modal
      onClose={onClose}
      open={open}
      size="lg"
      title={t("home.chooseThumbnail")}
    >
      {loading ? (
        <p className="m-0 text-caption text-secondary-text">{t("home.loadingSources")}</p>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          <button
            className="aspect-video overflow-hidden rounded-2xl border border-divider bg-black/30 p-0 text-left shadow-lg transition hover:-translate-y-0.5 hover:shadow-lg"
            onClick={() => void onSelect(null)}
            type="button"
          >
            <div className="flex h-full items-end bg-cover bg-center p-2">
              <span className="block truncate text-small font-semibold text-white">{t("home.noCover")}</span>
            </div>
          </button>
          {sources.map((source) => (
            <button
              className="aspect-video overflow-hidden rounded-2xl border border-divider bg-black/30 p-0 text-left shadow-lg transition hover:-translate-y-0.5 hover:shadow-lg"
              key={source.id}
              onClick={() => void onSelect(source.id)}
              type="button"
            >
              <div
                className="flex h-full items-end bg-cover bg-center p-2"
                style={{ backgroundImage: `linear-gradient(to top, rgba(0,0,0,0.5), transparent), url("/api/sources/${source.id}/cover")` }}
              >
                <span className="block truncate text-small font-semibold text-white">{source.title}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </Modal>
  );
}
