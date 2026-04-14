import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { IngestVideoIn, PreviewVideo, VideoStatus } from "@/lib/types";
import { formatDuration } from "@/lib/utils";
import { Modal } from "@/components/ui/Modal";
import { Spinner } from "@/components/ui/Spinner";
import { StatusChip } from "@/components/ui/StatusChip";
import { Thumbnail } from "@/components/ui/Thumbnail";
import { useLanguage } from "@/app/LanguageContext";

interface PlaylistPreviewModalProps {
  videos: PreviewVideo[];
  onSubmit: (selected: IngestVideoIn[]) => void;
  onCancel: () => void;
  submitting?: boolean;
  isLoading?: boolean;
}

const STATUS_LABEL_KEY: Record<Exclude<VideoStatus, "new">, string> = {
  processed: "lists.preview.status.processed",
  in_progress: "lists.preview.status.inProgress",
  needs_auth: "lists.preview.status.needsAuth",
};

const STATUS_CHIP_MAP: Record<Exclude<VideoStatus, "new">, "ok" | "error" | "unavailable" | "neutral"> = {
  processed: "ok",
  in_progress: "neutral",
  needs_auth: "error",
};

function VideoRow({
  video,
  selectable,
  selected,
  onToggle,
}: {
  video: PreviewVideo;
  selectable?: boolean;
  selected?: boolean;
  onToggle?: () => void;
}) {
  const { t } = useLanguage();
  const isProcessed = video.status !== "new";

  return (
    <div className={`flex items-center gap-3 ${isProcessed ? "opacity-60" : ""}`}>
      {selectable && (
        <input
          type="checkbox"
          className="h-4 w-4"
          checked={selected}
          onChange={onToggle}
          data-testid={`row-checkbox-${video.video_id}`}
        />
      )}
      <Thumbnail
        remoteUrl={video.cover_url}
        className="h-12 w-20 flex-shrink-0 rounded"
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium text-ink">{video.title}</span>
          {video.part_label && (
            <span className="inline-flex flex-shrink-0 rounded-full bg-blue/10 px-2 py-0.5 text-xs text-blue">
              {video.part_label}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-muted">
          <span>{video.uploader}</span>
          <span>·</span>
          <span>{formatDuration(video.duration_seconds)}</span>
          {isProcessed && (
            <StatusChip status={STATUS_CHIP_MAP[video.status as Exclude<VideoStatus, "new">]}>
              {t(STATUS_LABEL_KEY[video.status as Exclude<VideoStatus, "new">])}
            </StatusChip>
          )}
        </div>
      </div>
    </div>
  );
}

function VideoRowSkeleton({ selectable }: { selectable?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      {selectable && <div className="h-4 w-4 flex-shrink-0 rounded bg-border" />}
      <div className="h-12 w-20 flex-shrink-0 rounded bg-border animate-pulse" />
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="h-4 w-3/4 rounded bg-border animate-pulse" />
        <div className="h-3 w-1/2 rounded bg-border animate-pulse" />
      </div>
    </div>
  );
}

export function PlaylistPreviewModal({
  videos,
  onSubmit,
  onCancel,
  submitting = false,
  isLoading = false,
}: PlaylistPreviewModalProps) {
  const { t } = useLanguage();
  const newVideos = useMemo(() => videos.filter((v) => v.status === "new"), [videos]);
  const nonNewVideos = useMemo(() => videos.filter((v) => v.status !== "new"), [videos]);

  const [selected, setSelected] = useState<Set<string>>(() => new Set(newVideos.map((v) => v.video_id)));
  const masterRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setSelected(new Set(newVideos.map((v) => v.video_id)));
  }, [newVideos]);

  useEffect(() => {
    const input = masterRef.current;
    if (!input) return;
    if (selected.size === 0) {
      input.checked = false;
      input.indeterminate = false;
    } else if (selected.size === newVideos.length) {
      input.checked = true;
      input.indeterminate = false;
    } else {
      input.checked = false;
      input.indeterminate = true;
    }
  }, [selected, newVideos.length]);

  const handleMasterToggle = useCallback(() => {
    setSelected((prev) => {
      if (prev.size === newVideos.length) return new Set();
      return new Set(newVideos.map((v) => v.video_id));
    });
  }, [newVideos]);

  const handleRowToggle = useCallback((videoId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(videoId)) {
        next.delete(videoId);
      } else {
        next.add(videoId);
      }
      return next;
    });
  }, []);

  const handleSubmit = useCallback(() => {
    const selectedVideos = videos.filter((v) => v.status === "new" && selected.has(v.video_id));
    const payload: IngestVideoIn[] = selectedVideos.map(({ status, part_label, ...rest }) => rest);
    onSubmit(payload);
  }, [videos, selected, onSubmit]);

  const canSubmit = selected.size > 0 && !submitting && !isLoading;

  return (
    <Modal
      open={true}
      onClose={onCancel}
      title={t("lists.preview.title")}
      size="lg"
      footer={
        <>
          <button
            type="button"
            className="rounded-lg border border-border px-4 py-2 text-sm text-ink hover:bg-border/10"
            onClick={onCancel}
          >
            {t("lists.preview.cancel")}
          </button>
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-lg bg-blue px-4 py-2 text-sm text-white hover:bg-blue/90 disabled:opacity-50"
            onClick={handleSubmit}
            disabled={!canSubmit}
          >
            {submitting && <Spinner />}
            {t("lists.preview.submit")}
          </button>
        </>
      }
    >
      <div className="h-96 overflow-y-auto space-y-6">
        {newVideos.length > 0 && (
          <section>
            <div className="mb-3 flex items-center gap-3">
              <input
                ref={masterRef}
                type="checkbox"
                className="h-4 w-4"
                onChange={handleMasterToggle}
                aria-label={t("lists.preview.selectAll")}
                data-testid="master-checkbox"
              />
              <span className="text-sm font-medium text-ink">{t("lists.preview.selectAll")}</span>
              <span className="text-sm text-muted">({selected.size} / {newVideos.length})</span>
            </div>
            <div className="flex flex-col gap-3">
              {isLoading
                ? newVideos.map((v) => <VideoRowSkeleton key={v.video_id} selectable />)
                : newVideos.map((v) => (
                    <label key={v.video_id} className="cursor-pointer">
                      <VideoRow
                        video={v}
                        selectable
                        selected={selected.has(v.video_id)}
                        onToggle={() => handleRowToggle(v.video_id)}
                      />
                    </label>
                  ))}
            </div>
          </section>
        )}

        {nonNewVideos.length > 0 && (
          <section className="border-t border-border pt-6">
            <div className="mb-3 text-sm font-medium text-muted">
              {t("lists.preview.alreadyInList")}
            </div>
            <div className="flex flex-col gap-3">
              {isLoading
                ? nonNewVideos.map((v) => <VideoRowSkeleton key={v.video_id} />)
                : nonNewVideos.map((v) => <VideoRow key={v.video_id} video={v} />)}
            </div>
          </section>
        )}
      </div>
    </Modal>
  );
}
