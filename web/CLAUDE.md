# Web UI

React + TypeScript SPA. Managed with `npm`.

## Commands

```bash
npm install              # Install frontend dependencies
npm run dev              # Start Vite dev server on :5173
npm run build            # Production build to web/dist
npm run test             # Frontend test suite
npm run test -- list-detail-page   # Focused frontend tests
npm run lint             # Type-check the frontend
npx vitest run --coverage          # Coverage (requires @vitest/coverage-v8)
```

## Code Layout — `src/`

```
components/ui/    — primitive components (Button, Modal, Panel, Input, Select, SlotSlider, Spinner, StatusChip, SettingsField, Thumbnail, ContextMenu)
                    SlotSlider — 4-position segmented control (radiogroup, roving tabindex; keyboard nav moves focus + selection); used by LlmTab for context_window / max_output_tokens
components/auth/  — platform auth modals (BilibiliQrModal)
components/debug/ — prompt-trace reader: DebugDrawer + DebugHeader (opened from the assistant-bubble </> icon when debug_prompts + has_dump; Styled/Raw toggle over the per-turn dump)
components/*/     — feature components (lists/, lists/sources/, lists/lab/, lists/hooks/, jobs/, layout/, settings/)
                    lists/hooks/ holds chat hooks: useConversationHistory (history load + reattach eligibility via active_stream_message_id), useSSEStream (send/stop/retryMessage(assistantId)/reattach), useAutoScroll
                    lists/RetrievalLedger — structured ledger rendered above the assistant bubble; three variants (default/empty/pending)
pages/            — route-level page components
lib/              — typed api client, types, constants (SSE event types), artifact types, templates, download helpers, health check, i18n, utils
lib/hooks/        — useDebugDump (fetches GET /debug/messages/:msg_id for the prompt-trace drawer)
app/              — router, language context
test/             — Vitest test files + setup
```

## Routes

| Path | View |
|---|---|
| `/` | Home: grid of lists |
| `/lists/:id` | List detail: Sources \| Chat \| Lab |
| `/settings` | Global config, health, accounts |

## Conventions

- **Files**: `PascalCase` for components, `kebab-case` for utilities
- **Components**: props interface above component, `ComponentPropsWithoutRef<"tag">` for root element props, `Record<Variant, string>` for variant maps
- **Handlers**: named `handle{Action}`; event props use `on{Action}` prefix (`onDelete`, `onCreate`)
- **State**: `useState` with `set` prefix; async operations use `let cancelled = false` guard in `useEffect`
- **Imports**: use `@/*` alias (`@/components/ui`, `@/lib/api`, `@/lib/types`)
- **API client**: single `api` object in `lib/api.ts` with typed `request<T>` wrapper; errors thrown as `ApiError`. All HTTP requests must go through this client — do not use raw `fetch()` except for SSE streaming endpoints that require `ReadableStream` access.
- **Error handling**: always use `toErrorMessageWithT(error, t)` for user-facing error messages, never the raw `toErrorMessage()`. This ensures errors display in the correct UI language.
- **i18n**: `useLanguage()` → `t("key.path")` for lookup; `%{name}` placeholders with `t("key", { name: value })` for interpolation. String tables in `lib/i18n/{en,zh}.json` must stay in sync
- **Constants**: SSE event type strings go in `lib/constants.ts` (`meta`, `delta`, `citation`, `tool_call_start`, `tool_result`, `rag`, `done`, `error`, `cancelled`; `meta` is the stream-opener carrying `{message_id}`). Domain-specific event names (e.g., `BILIBILI_AUTH_REFRESH_EVENT`) live alongside their dispatcher in `lib/api.ts`. localStorage keys and utility constants live in `lib/utils.ts`. Never scatter the same magic string across multiple files.
- **Utilities**: before writing a helper function inside a component, check `lib/utils.ts` and `lib/chat-utils.ts` for existing implementations (`translateOrFallback`, `getErrorLabel`, `formatDurationHuman`, etc.). If a utility is pure (no React state), it belongs in `lib/`, not inline in a component.
- **Styling**: Tailwind utility classes only; no CSS modules. Inline `style` only for dynamic computed values (widths, positions, URLs). No arbitrary bracket values (e.g. `mt-[10px]`) — use Tailwind's built-in scale or CSS custom properties from `src/styles/app.css` (`--color-*`, `--z-*`, `--font-*`)
- **Cross-component auth sync**: When auth state changes (login/logout), call `notifyBilibiliAuthChanged()` from `lib/api.ts`. Components that need to react listen for `BILIBILI_AUTH_REFRESH_EVENT` via `window.addEventListener`. Do not prop-drill auth state through unrelated components.
- **RAG metadata**: The LLM dispatches one of two tool calls mid-stream — `find_passages` (recall locator, returns section-keyed excerpts + outline on facet match) or `read_section` (bounded verbatim read of one section by `[N]`). Results arrive as an SSE `tool_result` event with the corresponding `name`. Parsed in `useSSEStream` and surfaced via `ToolLedger` (rendered by `ToolLedgerRow`). On history reload, `useConversationHistory` reads `metadata.rag` from the stored message. Provisional-ledger pattern: `tool_call_start` pushes a `PendingRagCall` chip (collapsed, non-expandable; icon + i18n `summaryPending`/`readSectionPending` label) into `pendingRagCalls`; `tool_result` moves it to `rag.calls` (no `context[]` yet); the terminal `rag` event replaces with authoritative `rag.calls` whose `context[]` is reconstructed from the citation registry. While `msg.isStreaming`, `ToolLedger` rows render collapsed and non-expandable (`streaming` prop); post-stream they become expandable. `RetrievalCall`: `tool_name: "find_passages" | "read_section"`, `query` (`null` for `read_section`), `section_id?` + `source_id?` + `source_title?` (`read_section` only; the cited section + its source), `section_coverage: SectionCoverage[]` (narrowed to emitted citations; `find_passages` only), `context?: RetrievalChunk[]` (absent in the streaming `tool_result` payload, reconstructed in the terminal `rag` event; `[]` for `read_section`); v1 fields kept: `candidates_evaluated`, `sources_with_hits`, `sources_total`, `reranked`, `scoped_pool_size`, `facet_scope?`. v1 fields dropped: `mode`, `dropped_by_gate`, `gate_margin`, `video_id`. `SectionCoverage` is `{ section_id, source_id, source_title, seq, timestamp_start, timestamp_end }` (replaces the v1 `RagSource = { source_id, title }`). `RetrievalChunk` includes `citation_index`, `section_id`, `section_seq`, `source_id`, `source_title`, `timestamp_start`, `timestamp_end`. Citation SSE events carry `{ index, section_id, source_id, timestamp_start, chunk_ids }` — `section_id` + `timestamp_start` are the section-tab jump target the frontend threads through `onOpenSource` to `SourcesViewerMode`.
- **Reattach**: On mount, `useConversationHistory` reads `active_stream_message_id` from the conversation. If set, the `ChatPanel` calls `reattach()` which GETs `/api/lists/:id/chat/:msg_id/stream` and replays buffered SSE events, then tails live ones. Returns 204 after buffer eviction. The `cancelled` SSE event maps to `stoppedLabel`. The `error` event sets the message error field.
- **Cancel**: `stopStreaming()` POSTs to `/api/lists/:id/chat/:msg_id/cancel`. On non-2xx (e.g. 404 when the server hasn't persisted the message yet), falls back to client-side `AbortController.abort()`.

## Testing

- **Shared helpers, not inline mocks**: import from `@/test/utils` — `createMockApi`, `renderWithProviders`, `makeSseStream` / `makeOpenSseStream`, `mockFetch`. Don't re-roll per-file fetch stubs or hand-build `ApiClient` mocks.
- **API mocks via `createMockApi`**: inside a `vi.mock("@/lib/api", …)` factory, `const { createMockApi } = await import("@/test/utils")` (dynamic import dodges the hoist/circular-import trap), then return `{ api: createMockApi({ method: vi.fn()... }) }`. A compile-time drift guard fails `tsc` if `ApiClient` gains or loses a method without updating the factory's method list — keep them in sync.
- **Providers via `renderWithProviders(ui, { providers: [Outer, Inner] })`**: outermost-first; equivalent to `render(ui)` with none. Router-outer / Suspense-outer shapes needing a different nesting order render manually (documented exceptions).
- **SSE**: `makeSseStream(events)` for a closing stream; `makeOpenSseStream()` returns `{ response, enqueue, close }` to drive events mid-interaction (cancel-race tests).
- **fetch**: `mockFetch(handler)` wraps `vi.spyOn(window, "fetch")` so it's cleared by the test's `vi.restoreAllMocks()` — prefer it over `vi.stubGlobal("fetch", …)`; never assign `global.fetch` at module scope.
