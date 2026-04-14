import { Thumbnail } from "@/components/ui/Thumbnail";
import { formatDuration } from "@/lib/utils";

export function Banner({
  source,
  sourceUrl,
  uploader,
  durationSeconds,
}: {
  source: { id: string; cover_url: string | null };
  sourceUrl?: string;
  uploader: string;
  durationSeconds: number;
}) {
  const durationLabel = formatDuration(durationSeconds);

  const thumbnail = (
    <div className="relative h-64 w-full overflow-hidden rounded-xl bg-border">
      <Thumbnail
        source={source}
        alt={uploader}
        className="h-64 w-full"
      />
      {/* Gradient footer — floats over the bottom of the image */}
      <div className="absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-black/70 to-transparent" />
      {/* Uploader + duration overlay */}
      <div className="absolute inset-x-2 bottom-2 flex items-end justify-between">
        <span className="truncate text-xs font-medium text-white [text-shadow:0_1px_3px_rgba(0,0,0,0.6)]">
          {uploader}
        </span>
        <span className="shrink-0 rounded bg-black/50 px-1.5 py-0.5 text-xs font-medium text-white backdrop-blur-sm [text-shadow:0_1px_3px_rgba(0,0,0,0.6)]">
          {durationLabel}
        </span>
      </div>
    </div>
  );

  if (sourceUrl) {
    return (
      <a
        href={sourceUrl}
        rel="noopener noreferrer"
        target="_blank"
        className="block cursor-pointer"
      >
        {thumbnail}
      </a>
    );
  }

  return thumbnail;
}
