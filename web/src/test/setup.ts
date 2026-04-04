import "@testing-library/jest-dom/vitest";

// Mock ResizeObserver (not polyfilled by jsdom)
class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}
globalThis.ResizeObserver = MockResizeObserver;
