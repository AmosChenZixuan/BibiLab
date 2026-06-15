// Shared data-testid strings for the chat panel. Kept in lib/ per the project convention
// (web/CLAUDE.md "No magic strings") so component + test code share the same source of truth.
export const TEST_IDS = {
  bubbleUser: "bubble-user",
  bubbleAssistant: "bubble-assistant",
  citeChip: "cite-chip",
  citeChipTooltip: "cite-chip-tooltip",
  citeMissing: "cite-missing",
  digestKeywordChip: "digest-keyword-chip",
  debugDrawer: "debug-drawer",
} as const;
