import {
  LayoutDashboard,
  Activity,
  ScrollText,
  DatabaseZap,
  HeartPulse,
  Workflow,
  Route,
  Boxes,
  KeyRound,
  Plug,
  Wallet,
  Coins,
  PieChart,
  ShieldCheck,
  Bot,
  Globe,
  FileCheck2,
  Library,
  FlaskConical,
  TerminalSquare,
  BookOpen,
  History,
  Settings,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  label: string;
  path: string;
  icon: LucideIcon;
  /**
   * When true, the feature is part of Routeplane Enterprise and is NOT available
   * in the Community Edition. The nav still shows it (badged), but the page is a
   * static upgrade prompt — no enterprise code, endpoints, or data ship in CE.
   */
  enterprise?: boolean;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

// The full Console information architecture. Community-Edition features are wired
// to the local CE gateway; Enterprise-only features (`enterprise: true`) render an
// upgrade prompt. The IA mirrors the Enterprise Console so the shape is familiar.
export const NAV: NavGroup[] = [
  {
    title: "",
    items: [{ label: "Overview", path: "/", icon: LayoutDashboard }],
  },
  {
    title: "Observability",
    items: [
      { label: "Usage & Analytics", path: "/usage", icon: Activity },
      { label: "Logs & Traces", path: "/logs", icon: ScrollText },
      { label: "Cache", path: "/cache", icon: DatabaseZap },
      { label: "Provider Health", path: "/health", icon: HeartPulse },
    ],
  },
  {
    title: "Routing",
    items: [
      { label: "Smart Routing", path: "/routing/smart", icon: Workflow, enterprise: true },
      { label: "Routing Policies", path: "/routing/policies", icon: Route, enterprise: true },
      { label: "Model Catalog", path: "/models", icon: Boxes },
    ],
  },
  {
    title: "Keys & Providers",
    items: [
      { label: "API Keys", path: "/keys", icon: KeyRound },
      { label: "Provider Integrations", path: "/providers", icon: Plug },
    ],
  },
  {
    title: "FinOps",
    items: [
      { label: "Budgets & Limits", path: "/budgets", icon: Wallet },
      { label: "Smart Credits", path: "/credits", icon: Coins, enterprise: true },
      { label: "Cost Attribution", path: "/cost", icon: PieChart, enterprise: true },
    ],
  },
  {
    title: "Security & Compliance",
    items: [
      { label: "Guardrails", path: "/guardrails", icon: ShieldCheck, enterprise: true },
      { label: "Agentic Security", path: "/agents", icon: Bot, enterprise: true },
      { label: "Sovereignty & Residency", path: "/residency", icon: Globe, enterprise: true },
      { label: "Compliance & Artifacts", path: "/compliance", icon: FileCheck2, enterprise: true },
    ],
  },
  {
    title: "Prompts",
    items: [
      { label: "Registry", path: "/prompts", icon: Library, enterprise: true },
      { label: "Playground", path: "/playground", icon: FlaskConical },
    ],
  },
  {
    title: "Developer",
    items: [
      { label: "API Console", path: "/console", icon: TerminalSquare },
      { label: "API Reference", path: "/reference", icon: BookOpen },
      { label: "Admin Audit Log", path: "/audit", icon: History, enterprise: true },
    ],
  },
  {
    title: "",
    items: [{ label: "Settings", path: "/settings", icon: Settings }],
  },
];

/** Flat set of enterprise-only route paths (used by the router to gate pages). */
export const ENTERPRISE_PATHS = new Set(
  NAV.flatMap((g) => g.items).filter((i) => i.enterprise).map((i) => i.path),
);
