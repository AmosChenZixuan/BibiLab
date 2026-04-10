import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Copy } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { api } from "@/lib/api";
import type { Artifact } from "@/lib/types";

interface ArtifactViewerProps {
  artifact: Artifact;
}

export function ArtifactViewer({ artifact }: ArtifactViewerProps) {
  const { t } = useLanguage();
  const [content, setContent] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const result = await api.getArtifactContent(artifact.id);
        if (cancelled || !result) return;
        setContent(result.content);
      } catch {
        // Non-critical: content stays null
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [artifact.id]);

  function handleCopy() {
    if (!content) return;
    void navigator.clipboard.writeText(content);
  }

  const sourceCount = artifact.source_ids.length;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate font-serif text-base font-bold text-ink">{artifact.name}</p>
          <p className="m-0 text-xs text-muted">
            {t(sourceCount === 1 ? "lab.artifactViewer.basedOnSingular" : "lab.artifactViewer.basedOnPlural", { count: sourceCount })}
          </p>
        </div>
        <button
          type="button"
          aria-label="Copy markdown"
          disabled={!content}
          onClick={handleCopy}
          className="flex h-7 w-7 items-center justify-center rounded-full text-muted transition hover:bg-border hover:text-ink disabled:opacity-40"
        >
          <Copy size={14} />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {content ? (
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        ) : (
          <p className="text-sm text-muted">{t("lab.artifactViewer.loading")}</p>
        )}
      </div>
    </div>
  );
}
