import { createBrowserRouter, type RouteObject } from "react-router-dom";

import { AppFrame } from "@/components/layout/AppFrame";

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <AppFrame />,
    children: [
      {
        index: true,
        lazy: () => import("@/pages/HomePage").then((m) => ({ Component: m.HomePage })),
      },
      {
        path: "lists/:listId",
        lazy: () => import("@/pages/ListDetailPage").then((m) => ({ Component: m.ListDetailPage })),
      },
      {
        path: "settings",
        lazy: () => import("@/pages/SettingsPage").then((m) => ({ Component: m.SettingsPage })),
      },
    ],
  },
];

export const router = createBrowserRouter(routes);
