import { createBrowserRouter, type RouteObject } from "react-router-dom";

import { AppFrame } from "../components/layout/AppFrame";
import { HomePage } from "../pages/HomePage";
import { ListDetailPage } from "../pages/ListDetailPage";
import { SettingsPage } from "../pages/SettingsPage";
export const routes: RouteObject[] = [
  {
    path: "/",
    element: <AppFrame />,
    children: [
      { index: true, element: <HomePage /> },
      { path: "lists/:listId", element: <ListDetailPage /> },
      { path: "settings", element: <SettingsPage /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
