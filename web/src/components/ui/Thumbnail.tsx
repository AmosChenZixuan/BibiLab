import { ComponentPropsWithoutRef, useState } from "react";
import { Tv } from "lucide-react";

type SourceWithCover = { id: string; cover_url: string | null };

interface Props extends ComponentPropsWithoutRef<"img"> {
  source?: SourceWithCover;
  remoteUrl?: string | null;
}

export function Thumbnail({ source, remoteUrl, className = "", alt = "", ...rest }: Props) {
  const [loaded, setLoaded] = useState(false);
  const [fallbackUrl, setFallbackUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  const primaryUrl = source?.cover_url
    ? `/api/sources/${source.id}/cover`
    : remoteUrl && remoteUrl.trim()
      ? `/api/proxy/cover?url=${encodeURIComponent(remoteUrl)}`
      : "";

  const imgSrc = fallbackUrl || primaryUrl;
  const hasUrl = !!imgSrc && !failed;
  const showPlaceholder = !hasUrl || !loaded;

  function handleLoad(e: React.SyntheticEvent<HTMLImageElement>) {
    setLoaded(true);
    rest.onLoad?.(e);
  }

  function handleError(e: React.SyntheticEvent<HTMLImageElement>) {
    rest.onError?.(e);
    if (source?.cover_url && !fallbackUrl) {
      setFallbackUrl(`/api/proxy/cover?url=${encodeURIComponent(source.cover_url)}`);
    } else {
      setFailed(true);
    }
  }

  return (
    <div className={`relative bg-surface ${className}`} data-testid="thumbnail-wrapper">
      {hasUrl && (
        <img
          {...rest}
          src={imgSrc}
          alt={alt}
          loading="lazy"
          onLoad={handleLoad}
          onError={handleError}
          data-testid="thumbnail-img"
          className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
        />
      )}
      {showPlaceholder && (
        <div className={`absolute inset-0 flex items-center justify-center ${!loaded && hasUrl && !failed ? "animate-pulse" : ""}`} data-testid="thumbnail-placeholder">
          <Tv className="h-1/2 w-1/2 text-muted" />
        </div>
      )}
    </div>
  );
}
