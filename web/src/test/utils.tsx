// Shared test helpers for web/src/test/*. Test-only — never imported by
// production code. All exported names live under the @/test/utils alias.

import { render, type RenderResult } from "@testing-library/react";
import type { ComponentType, ReactElement, ReactNode } from "react";
import { type MockedFunction, type MockInstance, vi } from "vitest";

import { type ApiClient, api } from "@/lib/api";

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

/** Method names on a class-instance client (e.g. `AuthClient` methods live on
 *  the prototype, not own properties — `Object.keys` returns `[]`). */
function instanceMethodNames(obj: object): string[] {
  const proto = Object.getPrototypeOf(obj);
  return Object.getOwnPropertyNames(proto).filter(
    (n) =>
      n !== "constructor" &&
      typeof (obj as Record<string, unknown>)[n] === "function",
  );
}

/**
 * Build a fully-mocked `ApiClient` derived from the live `api` singleton. Every
 * method defaults to `vi.fn().mockResolvedValue(undefined)`, matching the real
 * `Promise<X | undefined>` signature so forgotten methods don't crash with
 * `TypeError: cannot read X of undefined`. Nested groups (currently `auth`)
 * are auto-mocked via a prototype walk.
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
  for (const key of Object.keys(api)) {
    const value = (api as unknown as Record<string, unknown>)[key];
    if (typeof value === "function") {
      mock[key] = vi.fn().mockResolvedValue(undefined);
    } else if (value !== null && typeof value === "object") {
      mock[key] = Object.fromEntries(
        instanceMethodNames(value).map((m) => [
          m,
          vi.fn().mockResolvedValue(undefined),
        ]),
      );
    }
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
