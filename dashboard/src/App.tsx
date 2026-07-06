import { lazy, Suspense, type ComponentType } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { Overview } from "@/routes/Overview";
import { SkeletonRows } from "@/components/ui/states";
import { Login } from "@/components/Login";
import { useAuthed } from "@/lib/auth";
import {
  SmartRouting,
  RoutingPolicies,
  Credits,
  Cost,
  Guardrails,
  Agents,
  Residency,
  Compliance,
  Prompts,
  AuditLog,
} from "@/routes/Enterprise";

// CE-functional pages are code-split; the tiny enterprise upsell stubs are
// imported eagerly (they're trivial).
const page = (loader: () => Promise<Record<string, ComponentType>>, name: string) =>
  lazy(() => loader().then((m) => ({ default: m[name] })));

const Usage = page(() => import("@/routes/Usage"), "Usage");
const Logs = page(() => import("@/routes/Logs"), "Logs");
const Cache = page(() => import("@/routes/Cache"), "Cache");
const Health = page(() => import("@/routes/Health"), "Health");
const Models = page(() => import("@/routes/Models"), "Models");
const Keys = page(() => import("@/routes/Keys"), "Keys");
const Providers = page(() => import("@/routes/Providers"), "Providers");
const Budgets = page(() => import("@/routes/Budgets"), "Budgets");
const Playground = page(() => import("@/routes/Playground"), "Playground");
const Reference = page(() => import("@/routes/Reference"), "Reference");
const Settings = page(() => import("@/routes/Settings"), "Settings");
const NotFound = page(() => import("@/routes/NotFound"), "NotFound");

const Fallback = () => (
  <div className="space-y-4">
    <SkeletonRows rows={1} className="max-w-xs" />
    <SkeletonRows rows={6} />
  </div>
);

const lazyRoute = (el: React.ReactNode) => <Suspense fallback={<Fallback />}>{el}</Suspense>;

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Overview /> },
      // Observability (CE)
      { path: "usage", element: lazyRoute(<Usage />) },
      { path: "logs", element: lazyRoute(<Logs />) },
      { path: "cache", element: lazyRoute(<Cache />) },
      { path: "health", element: lazyRoute(<Health />) },
      // Routing
      { path: "routing/smart", element: <SmartRouting /> },
      { path: "routing/policies", element: <RoutingPolicies /> },
      { path: "models", element: lazyRoute(<Models />) },
      // Keys & providers (CE, read-only)
      { path: "keys", element: lazyRoute(<Keys />) },
      { path: "providers", element: lazyRoute(<Providers />) },
      // FinOps
      { path: "budgets", element: lazyRoute(<Budgets />) },
      { path: "credits", element: <Credits /> },
      { path: "cost", element: <Cost /> },
      // Security & compliance (Enterprise)
      { path: "guardrails", element: <Guardrails /> },
      { path: "agents", element: <Agents /> },
      { path: "residency", element: <Residency /> },
      { path: "compliance", element: <Compliance /> },
      // Prompts
      { path: "prompts", element: <Prompts /> },
      { path: "playground", element: lazyRoute(<Playground />) },
      // Developer
      { path: "console", element: lazyRoute(<Playground />) },
      { path: "reference", element: lazyRoute(<Reference />) },
      { path: "audit", element: <AuditLog /> },
      // Settings (CE)
      { path: "settings", element: lazyRoute(<Settings />) },
      { path: "*", element: lazyRoute(<NotFound />) },
    ],
  },
]);

export function App() {
  const authed = useAuthed();
  if (!authed) return <Login />;
  return <RouterProvider router={router} />;
}
