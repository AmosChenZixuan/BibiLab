import { Search, BarChart3 } from "lucide-react";

export interface ToolDisplayConfig {
  icon: typeof Search;
  labelKey?: string;
  expand: boolean;
}

export const TOOL_DISPLAY: Record<string, ToolDisplayConfig> = {
  retrieve: { icon: Search, expand: true },
  survey: { icon: Search, expand: true },
  retrieve_scoped: { icon: Search, expand: true },
  query_list_metadata: { icon: BarChart3, labelKey: "chat.ledger.metadataLabel", expand: true },
};
