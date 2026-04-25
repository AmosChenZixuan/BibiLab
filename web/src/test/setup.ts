import "@testing-library/jest-dom/vitest";

// Mock ResizeObserver (not polyfilled by jsdom)
class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// Mock Element.scrollTo (not polyfilled by jsdom)
Element.prototype.scrollTo = function (
  optionsOrX?: ScrollToOptions | number,
  y?: number,
) {
  if (typeof optionsOrX === "number" && y !== undefined) {
    this.scrollLeft = optionsOrX;
    this.scrollTop = y;
  } else if (typeof optionsOrX === "object" && optionsOrX !== null) {
    this.scrollLeft = (optionsOrX as ScrollToOptions).left ?? this.scrollLeft;
    this.scrollTop = (optionsOrX as ScrollToOptions).top ?? this.scrollTop;
  }
};
