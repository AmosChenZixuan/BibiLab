import "@testing-library/jest-dom/vitest";

// Mock ResizeObserver (not polyfilled by jsdom)
class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// Mock Element.scrollTo (not polyfilled by jsdom)
Element.prototype.scrollTo = function (options?: ScrollToOptions | number) {
  if (typeof options === "number") {
    this.scrollTop = options;
  } else if (options) {
    this.scrollTop = (options as ScrollToOptions).top ?? this.scrollTop;
  }
};
