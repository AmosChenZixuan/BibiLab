import { useEffect, useRef } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";
import { formatDurationHuman } from "@/lib/chat-utils";
import type { SourceSection } from "@/lib/types";

const TABS_VISIBLE = 3;

interface Props {
  sections: SourceSection[];
  activeIdx: number;
  onActiveIdxChange: (idx: number) => void;
}

/** Paged tab strip. Exactly 3 tabs are visible at any time. ‹ / › arrows
 *  slide the window by one tab. Active tab auto-centers when possible.
 *  When `sections.length <= 3`, arrows are hidden and all tabs are visible.
 *
 *  Controlled: parent owns `activeIdx`. PagerTabs only renders the visual
 *  state and reports clicks via `onActiveIdxChange`. This keeps a single
 *  source of truth for the active index and prevents the tab/body desync
 *  that arises when the parent holds its own copy (e.g., when sections
 *  are briefly cleared between source switches). */
export function PagerTabs({ sections, activeIdx, onActiveIdxChange }: Props) {
  const { t } = useLanguage();
  const activeTabRef = useRef<HTMLButtonElement>(null);

  // After the active tab changes, scroll it into the center slot of the
  // 3-tab visible window. Guarded for jsdom where `scrollIntoView`
  // doesn't exist.
  useEffect(() => {
    const el = activeTabRef.current;
    if (el && typeof el.scrollIntoView === "function") {
      el.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
    }
  }, [activeIdx]);

  const canPrev = activeIdx > 0;
  const canNext = activeIdx < sections.length - 1;
  const showArrows = sections.length > TABS_VISIBLE;

  return (
    <div className="flex items-stretch border-b border-border">
      {showArrows && (
        <button
          type="button"
          aria-label={t("lists.sections.previousSection")}
          onClick={() => canPrev && onActiveIdxChange(activeIdx - 1)}
          disabled={!canPrev}
          className={`flex w-7 shrink-0 items-center justify-center bg-transparent text-blue ${
            canPrev ? "hover:bg-blue/5" : "cursor-not-allowed text-muted opacity-30"
          }`}
        >
          <ChevronLeft size={14} />
        </button>
      )}

      <div className="min-w-0 flex-1 overflow-hidden">
        <div
          role="tablist"
          aria-label={t("lists.sections.label")}
          className="flex snap-x snap-mandatory overflow-x-auto scrollbar-none"
          style={{ scrollbarWidth: "none" }}
        >
          {sections.map((s, i) => (
            <button
              key={s.seq}
              ref={i === activeIdx ? activeTabRef : undefined}
              type="button"
              role="tab"
              aria-selected={i === activeIdx}
              onClick={() => onActiveIdxChange(i)}
              className={`shrink-0 snap-center border-b-2 border-transparent bg-transparent px-3 py-2 font-mono text-xs tracking-normal text-muted transition-colors hover:text-ink ${
                i === activeIdx ? "border-blue font-semibold text-blue" : ""
              }`}
              style={{ flex: "0 0 calc(100% / 3)" }}
              title={`${formatDurationHuman(s.timestamp_start)} – ${formatDurationHuman(s.timestamp_end)}`}
            >
              {formatDurationHuman(s.timestamp_start)} – {formatDurationHuman(s.timestamp_end)}
            </button>
          ))}
        </div>
      </div>

      {showArrows && (
        <button
          type="button"
          aria-label={t("lists.sections.nextSection")}
          onClick={() => canNext && onActiveIdxChange(activeIdx + 1)}
          disabled={!canNext}
          className={`flex w-7 shrink-0 items-center justify-center bg-transparent text-blue ${
            canNext ? "hover:bg-blue/5" : "cursor-not-allowed text-muted opacity-30"
          }`}
        >
          <ChevronRight size={14} />
        </button>
      )}
    </div>
  );
}
