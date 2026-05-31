import { ComponentPropsWithoutRef, useState } from "react";
import { Tv } from "lucide-react";

interface Props extends Omit<ComponentPropsWithoutRef<"img">, "src"> {
  src?: string | null;
  /** Tried once if `src` fails to load, before falling back to the placeholder. */
  fallbackSrc?: string | null;
}

export function Thumbnail({ src, fallbackSrc, className = "", alt = "", ...rest }: Props) {
  const [loaded, setLoaded] = useState(false);
  const [usingFallback, setUsingFallback] = useState(false);
  const [failed, setFailed] = useState(false);

  const imgSrc = usingFallback ? fallbackSrc : src;
  const hasUrl = !!imgSrc && !failed;
  const showPlaceholder = !hasUrl || !loaded;

  function handleLoad(e: React.SyntheticEvent<HTMLImageElement>) {
    setLoaded(true);
    rest.onLoad?.(e);
  }

  function handleError(e: React.SyntheticEvent<HTMLImageElement>) {
    rest.onError?.(e);
    if (!usingFallback && fallbackSrc) {
      setUsingFallback(true);
    } else {
      setFailed(true);
    }
  }

  return (
    <div className={`relative bg-surface ${className}`} data-testid="thumbnail-wrapper">
      {hasUrl && (
        <img
          {...rest}
          src={imgSrc ?? undefined}
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
