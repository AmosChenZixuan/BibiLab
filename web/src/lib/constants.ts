export const CHAT_MODE_FOCUSED = "focused" as const;
export const CHAT_MODE_BROAD = "broad" as const;
export type ChatMode = typeof CHAT_MODE_FOCUSED | typeof CHAT_MODE_BROAD;
