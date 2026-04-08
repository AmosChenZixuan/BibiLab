import { describe, expect, test } from "vitest";

import { routes } from "@/app/routes";

describe("lazy routes", () => {
  test("page components use React Router's lazy property for code splitting", () => {
    // Verify that routes use React Router's built-in lazy property for code splitting
    // This ensures pages are code-split and loaded on demand without needing React.lazy + Suspense
    const listDetailRoute = routes[0].children?.find((r: { path?: string }) => r.path === "lists/:listId");
    expect(listDetailRoute).toBeDefined();

    // React Router's lazy loading uses the `lazy` property on routes
    // which accepts a function that returns a promise for the route module
    expect(listDetailRoute).toHaveProperty("lazy");
    expect(typeof (listDetailRoute as { lazy: unknown }).lazy).toBe("function");
  });

  test("all page routes have lazy property for code splitting", () => {
    // All page routes should be lazy-loaded
    const pageRoutes = routes[0].children;
    expect(pageRoutes).toBeDefined();
    expect(pageRoutes!).toHaveLength(3); // HomePage, ListDetailPage, SettingsPage

    for (const route of pageRoutes!) {
      const r = route as { lazy?: unknown };
      expect(r).toHaveProperty("lazy");
      expect(typeof r.lazy).toBe("function");
    }
  });
});
