import { Search, BookOpen } from "lucide-react";

export const FIND_PASSAGES_TOOL_NAME = "find_passages";
export const READ_SOURCE_TOOL_NAME = "read_source";

export interface ToolDisplayConfig {
  icon: typeof Search;
  labelKey?: string;
}

export const TOOL_DISPLAY: Record<string, ToolDisplayConfig> = {
  [FIND_PASSAGES_TOOL_NAME]: { icon: Search },
  [READ_SOURCE_TOOL_NAME]: { icon: BookOpen, labelKey: "chat.ledger.readSourceLabel" },
};
