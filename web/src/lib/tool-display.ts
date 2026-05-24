import { Search, BarChart3 } from "lucide-react";

export const METADATA_TOOL_NAME = "query_list_metadata";

export interface ToolDisplayConfig {
  icon: typeof Search;
  labelKey?: string;
}

export const TOOL_DISPLAY: Record<string, ToolDisplayConfig> = {
  retrieve: { icon: Search },
  survey: { icon: Search },
  retrieve_scoped: { icon: Search },
  [METADATA_TOOL_NAME]: { icon: BarChart3, labelKey: "chat.ledger.metadataLabel" },
};
