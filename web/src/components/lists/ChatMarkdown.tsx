import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Root, Element, Text, RootContent, Properties } from "hast";
import type { ReactNode } from "react";

import { useLanguage } from "@/app/LanguageContext";
import { TEST_IDS } from "@/lib/test-ids";
import type { Source } from "@/lib/types";
import type { ContentBlock, OpenSourceOpts } from "@/lib/chat-utils";

function CitationChip({
  index,
  sourceId,
  chunkIds,
  sectionId,
  timestampStart,
  sources,
  onOpenSource,
}: {
  index: number;
  sourceId: string;
  chunkIds: string[];
  sectionId?: string;
  timestampStart?: number;
  sources: Source[];
  onOpenSource?: (source: Source, opts?: OpenSourceOpts) => void;
}) {
  const { t } = useLanguage();
  const source = sources.find((s) => s.id === sourceId);
  if (!source) {
    return (
      <span
        className="text-2xs text-muted cursor-not-allowed opacity-60"
        data-testid={TEST_IDS.citeMissing}
        title={t("chat.citationMissing")}
      >
        [{index}]
      </span>
    );
  }
  return (
    <span className="group/cite relative inline">
      <button
        type="button"
        onClick={() => onOpenSource?.(source, {
          highlightChunks: chunkIds,
          sectionId,
          timestampStart,
        })}
        data-testid={TEST_IDS.citeChip}
        className="mx-px border-0 bg-transparent p-0 text-2xs font-semibold text-blue cursor-pointer hover:underline focus-visible:rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue focus-visible:outline-offset-1"
      >
        [{index}]
      </button>
      <span
        data-testid={TEST_IDS.citeChipTooltip}
        className="pointer-events-none absolute bottom-full left-1/2 z-50 hidden w-max max-w-64 -translate-x-1/2 -translate-y-1.5 rounded-md bg-ink px-2 py-1 text-xs leading-snug text-white break-words group-hover/cite:block group-focus-within/cite:block"
      >
        {source.title}
      </span>
    </span>
  );
}


export const CITE_TOKEN_RE = /​⁣CITE(\d+)⁣​/;

function makeCiteToken(idx: number): string {
  // U+200B (ZWSP) + U+2063 (invisible separator) — both zero-width,
  // survive markdown parsing as inline text, won't be stripped by formatters.
  return `​⁣CITE${idx}⁣​`;
}

type CiteData = {
  index: number;
  source_id: string;
  chunk_ids: string[];
  section_id?: string;
  timestamp_start?: number;
  sources: Source[];
  onOpenSource?: (source: Source, opts?: OpenSourceOpts) => void;
};

function CiteEl(props: { _cite?: CiteData }) {
  const cite = props._cite;
  if (!cite) {
    console.warn("CiteEl: missing _cite prop — possible citation index mismatch");
    return null;
  }
  return (
    <CitationChip
      index={cite.index}
      sourceId={cite.source_id}
      chunkIds={cite.chunk_ids}
      sectionId={cite.section_id}
      timestampStart={cite.timestamp_start}
      sources={cite.sources}
      onOpenSource={cite.onOpenSource}
    />
  );
}

const MARKDOWN_COMPONENTS = {
  p: ({ children }: { children?: ReactNode }) => <>{children}</>,
  // react-markdown's Components type only knows intrinsic HTML tags; the custom
  // citation-el element (injected by makeRehypeCitePlugin) needs the assertion.
  "citation-el": CiteEl,
} as Components;

function makeRehypeCitePlugin(citations: CiteData[]) {
  // Attacher called by unified.use(); returns the actual transformer.
  return function rehypeCiteTokens(): (tree: Root) => void {
    return function transform(tree: Root): void {
      walk(tree);

      function walk(node: Root | Element): void {
        if (!node.children) return;
        for (let i = node.children.length - 1; i >= 0; i--) {
          const child = node.children[i];
          if (child.type === "text") {
            if (!CITE_TOKEN_RE.test(child.value)) continue;
            const parts = child.value.split(CITE_TOKEN_RE);
            const replacements: (Text | Element)[] = [];
            for (let j = 0; j < parts.length; j++) {
              if (j % 2 === 0) {
                if (parts[j]) replacements.push({ type: "text", value: parts[j] });
              } else {
                const idx = Number(parts[j]);
                if (!citations[idx]) {
                  console.warn("rehypeCiteTokens: cite token out of bounds", parts[j], citations.length);
                  continue;
                }
                replacements.push({
                  type: "element",
                  tagName: "citation-el",
                  // hast property values are primitives; the CiteData object
                  // rides through as an opaque prop to CiteEl, hence the cast.
                  properties: { _cite: citations[idx] } as unknown as Properties,
                  children: [],
                });
              }
            }
            node.children.splice(i, 1, ...replacements as RootContent[]);
          } else if (child.type === "element") {
            walk(child);
          }
        }
      }
    };
  };
}

export function renderParagraphs(
  contentBlocks: ContentBlock[],
  sources: Source[],
  onOpenSource?: (source: Source, opts?: OpenSourceOpts) => void,
  isStreaming?: boolean,
) {
  // Split into paragraphs on paragraph_break
  const paragraphs: Array<Array<ContentBlock>> = [[]];
  const last = () => paragraphs[paragraphs.length - 1];
  for (const block of contentBlocks) {
    if (block.type === "paragraph_break") {
      if (last().length > 0) {
        paragraphs.push([]);
      }
    } else {
      last().push(block);
    }
  }

  // Post-merge fold: citation-only trailing paragraphs attach to previous
  for (let i = paragraphs.length - 1; i > 0; i--) {
    if (paragraphs[i].length > 0 && paragraphs[i].every((b) => b.type === "citation")) {
      paragraphs[i - 1].push(...paragraphs[i]);
      paragraphs[i] = [];
    }
  }

  return (
    <>
      {paragraphs.map((para, pi) => {
        if (para.length === 0) return null;

        const citations: CiteData[] = [];
        let merged = "";
        for (const block of para) {
          if (block.type === "text") {
            merged += block.text;
          } else if (block.type === "citation") {
            merged += makeCiteToken(citations.length);
            citations.push({
              index: block.index,
              source_id: block.source_id,
              chunk_ids: block.chunk_ids,
              section_id: block.section_id,
              timestamp_start: block.timestamp_start,
              sources,
              onOpenSource,
            });
          }
        }

        return (
          <div key={pi} className="citation-paragraph">
            <ReactMarkdown
              components={MARKDOWN_COMPONENTS}
              rehypePlugins={[makeRehypeCitePlugin(citations)]}
              remarkPlugins={[remarkGfm]}
            >
              {merged}
            </ReactMarkdown>
          </div>
        );
      })}
      {isStreaming && (
        <span className="inline-block w-0.5 h-3.5 bg-blue align-text-bottom ml-0.5 chat-cursor-blink" />
      )}
    </>
  );
}
