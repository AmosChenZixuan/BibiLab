// Shared test helpers for web/src/test/*. Test-only — never imported by
// production code. All exported names live under the @/test/utils alias.

import { render, type RenderResult } from "@testing-library/react";
import type { ComponentType, ReactElement, ReactNode } from "react";
import { type MockedFunction, type MockInstance, vi } from "vitest";

import type { ApiClient } from "@/lib/api";

// ─── Types ───────────────────────────────────────────────────────────────────

/** Wrap a method signature in vitest's `MockedFunction` (preserves the call
 *  signature and exposes the mock API). */
type MockedMethod<T> = T extends (...args: infer A) => infer R
  ? MockedFunction<(...args: A) => R>
  : T;

/** Fully-mocked `ApiClient` — every method is a `MockedFunction` of the real
 *  signature, so `mockedApi.getHealth.mockResolvedValue(...)` is type-checked
 *  against the real return type. The nested `auth` group is also mocked. */
export type MockedApiClient = {
  [K in keyof ApiClient]: K extends "auth"
    ? { [NK in keyof ApiClient[K]]: MockedMethod<ApiClient[K][NK]> }
    : MockedMethod<ApiClient[K]>;
};

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
