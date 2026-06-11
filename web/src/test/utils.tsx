// Shared test helpers for web/src/test/*. Test-only — never imported by
// production code. All exported names live under the @/test/utils alias.

import { render, type RenderResult } from "@testing-library/react";
import type { ComponentType, ReactElement, ReactNode } from "react";
import { type MockedFunction, type MockInstance, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { RetrievalCall } from "@/lib/chat-utils";
import type { Source, SourceSection } from "@/lib/types";

// ─── Types ───────────────────────────────────────────────────────────────────

/** Wrap a method signature in vitest's `MockedFunction` (preserves the call
 *  signature and exposes the mock API). */
type MockedMethod<T> = T extends (...args: infer A) => infer R
  ? MockedFunction<(...args: A) => R>
  : T;

/** Fully-mocked `ApiClient` — every method is a `MockedFunction` of the real
 *  signature, so `mockedApi.getHealth.mockResolvedValue(...)` is type-checked
 *  against the real return type. The nested `auth` group is also mocked. */
type MockedApiClient = {
  [K in keyof ApiClient]: K extends "auth"
    ? { [NK in keyof ApiClient[K]]: MockedMethod<ApiClient[K][NK]> }
    : MockedMethod<ApiClient[K]>;
};

// ─── Shared fixtures ────────────────────────────────────────────────────────

/** Canonical source fixture used by the chat-panel tests. */
export const SOURCE_1: Source = {
  id: "src-1",
  video_id: "BV1test",
  platform: "bilibili",
  title: "Test Video A",
  cover_url: null,
  source_url: "https://bilibili.com/video/BV1test",
  duration_seconds: 3600,
  uploader: "TestUploader",
  language: "en",
  processed_at: "2026-04-08T12:00:00Z",
};

/** Second source fixture for tests that need two sources. */
export const SOURCE_2: Source = {
  id: "src-2",
  video_id: "BV1test2",
  platform: "bilibili",
  title: "Test Video B",
  cover_url: null,
  source_url: "https://bilibili.com/video/BV1test2",
  duration_seconds: 1800,
  uploader: "TestUploader",
  language: "en",
  processed_at: "2026-04-08T13:00:00Z",
};

// ─── Retrieval fixtures ─────────────────────────────────────────────────────

/** Canonical find_passages `RetrievalCall` — shared by the ToolLedger and
 *  ToolLedgerRow test suites. Mutate via spread at the call site:
 *  `const streaming = { ...MOCK_RETRIEVAL_CALL, query: "..." };`. */
export const MOCK_RETRIEVAL_CALL: RetrievalCall = {
  query: "长期情景记忆",
  tool_name: "find_passages",
  candidates_evaluated: 10,
  sources_with_hits: 1,
  sources_total: 16,
  section_coverage: [
    { section_id: "sec1", source_id: "s1", source_title: "Test Video", seq: 1, timestamp_start: 0, timestamp_end: 132 },
    { section_id: "sec2", source_id: "s1", source_title: "Test Video", seq: 2, timestamp_start: 132, timestamp_end: 300 },
  ],
  context: [
    {
      chunk_id: "c1",
      citation_index: 1,
      section_id: "sec1",
      section_seq: 1,
      source_id: "s1",
      source_title: "Test Video",
      timestamp_start: 0,
      timestamp_end: 132,
      rerank_score: 4.53,
      preview: "面试官问在构建一个长期陪伴性AI角色时 如何设计…",
    },
    {
      chunk_id: "c2",
      citation_index: 2,
      section_id: "sec2",
      section_seq: 2,
      source_id: "s1",
      source_title: "Test Video",
      timestamp_start: 132,
      timestamp_end: 300,
      rerank_score: 3.21,
      preview: "Another preview text",
    },
  ],
  reranked: true,
  scoped_pool_size: 10,
};

// ─── Section fixtures ────────────────────────────────────────────────────────

/**
 * Build a list of `SourceSection` fixtures. Defaults match the
 * SourcesViewerMode test style (1 keyword per section, `summary` text
 * `Section N summary`). Pass `summarySuffix` (e.g. " text") and/or
 * `keywordsPerSection: 2` for the DigestAccordion test style.
 *
 * - `idOffset` shifts the section_id sequence (e.g. idOffset=100 → `sec-101`).
 * - `startSec` shifts the timestamp window (defaults to 0; the
 *   SourcesViewerMode test ignores this and uses raw `i * 600`).
 */
export function makeSections(
  n: number,
  opts: {
    idOffset?: number;
    startSec?: number;
    keywordsPerSection?: number;
    summarySuffix?: string;
  } = {},
): SourceSection[] {
  const { idOffset = 0, startSec = 0, keywordsPerSection = 1, summarySuffix = "" } = opts;
  return Array.from({ length: n }, (_, i) => {
    const idx = idOffset + i + 1;
    return {
      section_id: `sec-${idx}`,
      seq: i + 1,
      summary: `Section ${idx} summary${summarySuffix}`,
      keywords: Array.from(
        { length: keywordsPerSection },
        (_, k) => `kw-${idx}${keywordsPerSection > 1 ? String.fromCharCode(97 + k) : ""}`,
      ),
      timestamp_start: startSec + i * 600,
      timestamp_end: startSec + (i + 1) * 600,
    };
  });
}

// ─── createMockApi ───────────────────────────────────────────────────────────

/**
 * Top-level ApiClient method names — keep in sync with the `ApiClient`
 * interface in `@/lib/api`. Hard-coded so the factory can be used inside
 * `vi.mock` factories without triggering a circular import through
 * `@/test/utils` → `@/lib/api`. Add a method here when adding one to
 * `ApiClient`.
 */
const TOP_LEVEL_METHODS = [
  "listLists",
  "createList",
  "updateList",
  "deleteList",
  "createArtifact",
  "listSources",
  "getSource",
  "deleteSource",
  "rerunDigest",
  "updateSourceFacets",
  "getSourceSections",
  "previewPlaylist",
  "previewPlaylistMetadata",
  "ingestUrl",
  "listArtifacts",
  "getArtifactContent",
  "updateArtifact",
  "deleteArtifact",
  "getConfig",
  "putConfig",
  "getHealth",
  "listJobs",
  "deleteJob",
  "listModels",
  "downloadModel",
  "syncModels",
  "getConversation",
  "deleteConversation",
  "getDebugDump",
] as const;

/** `auth` group method names — keep in sync with `ApiClient.auth`. */
const AUTH_METHODS = [
  "generateBilibiliQr",
  "pollBilibiliQr",
  "deleteBilibiliAuth",
] as const;

// Compile-time drift guard: if a method is added to or removed from
// `ApiClient` / `ApiClient.auth` without updating the lists above, the
// `_TopLevelDriftCheck` / `_AuthDriftCheck` types evaluate to `never` and
// the `const _drift: ... = true` line fails to typecheck.
type _Exclude<T, U> = T extends U ? never : T;
type _TopLevelDriftCheck = _Exclude<
  Exclude<keyof ApiClient, "auth">,
  (typeof TOP_LEVEL_METHODS)[number]
> extends never
  ? _Exclude<
      (typeof TOP_LEVEL_METHODS)[number],
      Exclude<keyof ApiClient, "auth">
    > extends never
    ? true
    : `TOP_LEVEL_METHODS has a name not in ApiClient`
  : `ApiClient has a top-level method not in TOP_LEVEL_METHODS`;
type _AuthDriftCheck = _Exclude<
  keyof ApiClient["auth"],
  (typeof AUTH_METHODS)[number]
> extends never
  ? _Exclude<(typeof AUTH_METHODS)[number], keyof ApiClient["auth"]> extends never
    ? true
    : `AUTH_METHODS has a name not in ApiClient.auth`
  : `ApiClient.auth has a method not in AUTH_METHODS`;
const _topLevelDrift: _TopLevelDriftCheck = true;
const _authDrift: _AuthDriftCheck = true;
void _topLevelDrift;
void _authDrift;

/**
 * Build a fully-mocked `ApiClient`. Every method defaults to
 * `vi.fn().mockResolvedValue(undefined)`, matching the real
 * `Promise<X | undefined>` signature so forgotten methods don't crash with
 * `TypeError: cannot read X of undefined`. The `auth` nested group is also
 * auto-mocked.
 *
 * Pass `overrides` to replace specific methods. Top-level method overrides
 * replace wholesale; nested-group overrides shallow-merge so you can override
 * a single auth method without re-declaring the rest.
 *
 * @example
 *   createMockApi({ getHealth: vi.fn().mockResolvedValue({ ok: true }) })
 *   createMockApi({ auth: { generateBilibiliQr: vi.fn().mockResolvedValue(...) } })
 */
export function createMockApi(
  overrides: Partial<MockedApiClient> = {},
): MockedApiClient {
  const mock: Record<string, unknown> = {};
  for (const key of TOP_LEVEL_METHODS) {
    mock[key] = vi.fn().mockResolvedValue(undefined);
  }
  mock.auth = {};
  for (const key of AUTH_METHODS) {
    (mock.auth as Record<string, unknown>)[key] = vi
      .fn()
      .mockResolvedValue(undefined);
  }
  for (const [k, v] of Object.entries(overrides)) {
    if (
      v !== null &&
      typeof v === "object" &&
      !Array.isArray(v) &&
      mock[k] !== null &&
      typeof mock[k] === "object" &&
      !Array.isArray(mock[k])
    ) {
      // Nested group: shallow-merge so partial overrides stick.
      mock[k] = { ...(mock[k] as object), ...v };
    } else {
      // Top-level method: replace wholesale.
      mock[k] = v;
    }
  }
  return mock as unknown as MockedApiClient;
}

// ─── renderWithProviders ─────────────────────────────────────────────────────

/**
 * Render `ui` inside a stack of provider components. The caller specifies
 * order via the `providers` array (outermost first). With no providers, this
 * is equivalent to `render(ui)`.
 *
 * @example
 *   renderWithProviders(<MyComponent />, {
 *     providers: [LanguageProvider, JobActivityProvider],
 *   })
 */
export function renderWithProviders(
  ui: ReactElement,
  {
    providers = [],
  }: { providers?: ComponentType<{ children: ReactNode }>[] } = {},
): RenderResult {
  return render(
    providers.reduceRight<ReactElement>(
      (acc, Provider) => <Provider>{acc}</Provider>,
      ui,
    ),
  );
}

// ─── SSE helpers ─────────────────────────────────────────────────────────────

const SSE_HEADERS = { "Content-Type": "text/event-stream" } as const;
const textEncoder = new TextEncoder();

/**
 * Build a `Response` whose body is an SSE stream that enqueues each event
 * string verbatim, then closes. Use for tests that don't need to interleave
 * events with user actions.
 */
export function makeSseStream(events: string[]): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const event of events) {
          controller.enqueue(textEncoder.encode(event));
        }
        controller.close();
      },
    }),
    { headers: SSE_HEADERS },
  );
}

/**
 * Build an SSE stream that stays open. Returns the response plus `enqueue` /
 * `close` handles so the test can drive events mid-interaction (e.g., emit a
 * `tool_call_start` before the user has finished typing, or interleave
 * `tool_result` with `delta` events). This is the cancel-race helper.
 */
export function makeOpenSseStream(): {
  response: Response;
  enqueue: (chunk: string) => void;
  close: () => void;
} {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
    },
  });
  return {
    response: new Response(body, { headers: SSE_HEADERS }),
    enqueue: (chunk) => controller.enqueue(textEncoder.encode(chunk)),
    close: () => controller.close(),
  };
}

// ─── mockFetch ───────────────────────────────────────────────────────────────

/**
 * Thin wrapper around `vi.spyOn(window, "fetch").mockImplementation(handler)`.
 * The spy is restored by the test's existing `vi.restoreAllMocks()` in
 * afterEach. Use this in place of `vi.stubGlobal("fetch", …)` so the fetch
 * mock follows the same lifecycle as other spy mocks in the test.
 */
export function mockFetch(
  handler: (
    input: RequestInfo | URL,
    init?: RequestInit,
  ) => Promise<Response>,
): MockInstance {
  return vi.spyOn(window, "fetch").mockImplementation(handler);
}

// ─── renderChatPanel ─────────────────────────────────────────────────────────

/**
 * Render a `ChatPanel` (or any compatible component) inside a provider
 * stack. The default props match the minimum mount surface used by the
 * SSE and RAG-observability tests; pass overrides to select sources,
 * change `listId`, or supply `onOpenSource`. Tests with a different
 * signature (e.g. `chat-panel.test.tsx`'s `{ skipMock }`) keep their
 * own renderer.
 *
 * Both the component and the providers are passed in (rather than
 * imported here) so this module has no runtime dependency on either
 * `ChatPanel` or the providers. This is load-bearing: tests that only
 * use `createMockApi` (e.g. `useDebugDump.test.tsx`) load `utils.tsx`
 * without pulling in the providers' transitive `@/lib/api` import —
 * which would otherwise defeat the `vi.mock("@/lib/api", …)` mock when
 * this file is dynamically imported from inside the mock factory.
 */
export function renderChatPanel(
  Component: ComponentType<{
    selectedSourceIds: string[];
    sources: Source[];
    listId: string;
    onOpenSource?: (
      source: Source,
      opts?: { sectionId?: string; timestampStart?: number; highlightChunks?: string[] },
    ) => void;
  }>,
  options: {
    providers?: ComponentType<{ children: ReactNode }>[];
  } & Partial<{
    selectedSourceIds: string[];
    sources: Source[];
    listId: string;
    onOpenSource?: (
      source: Source,
      opts?: { sectionId?: string; timestampStart?: number; highlightChunks?: string[] },
    ) => void;
  }> = {},
): RenderResult {
  const { providers, ...props } = options;
  return renderWithProviders(
    <Component
      selectedSourceIds={[]}
      sources={[]}
      listId="list-1"
      {...props}
    />,
    { providers: providers ?? [] },
  );
}
