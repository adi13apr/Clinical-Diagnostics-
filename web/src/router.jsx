import { createBrowserRouter } from "react-router";
import { AppShell } from "@/components/layout/AppShell";
import { RequireAuth } from "@/components/layout/RequireAuth";
import { DiagnosePage } from "@/pages/DiagnosePage";
import { LoginPage } from "@/pages/LoginPage";
import { NotFoundPage, PageLoading, RouteErrorPage } from "@/pages/NotFoundPage";

// Charts and tables load on first visit, keeping the first page light
const lazyPage = (load, name) => async () => ({ Component: (await load())[name] });

export const router = createBrowserRouter([
  { path: "/login", Component: LoginPage, ErrorBoundary: RouteErrorPage },
  {
    Component: RequireAuth,
    ErrorBoundary: RouteErrorPage,
    HydrateFallback: PageLoading, // shown while a lazy page loads on a direct visit
    children: [
      {
        Component: AppShell,
        children: [
          { index: true, Component: DiagnosePage },
          { path: "history", lazy: lazyPage(() => import("@/pages/HistoryPage"), "HistoryPage") },
          { path: "analytics", lazy: lazyPage(() => import("@/pages/AnalyticsPage"), "AnalyticsPage") },
          { path: "traces/:traceId?", lazy: lazyPage(() => import("@/pages/TracePage"), "TracePage") },
          { path: "*", Component: NotFoundPage },
        ],
      },
    ],
  },
]);
