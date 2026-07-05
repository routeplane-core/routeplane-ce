import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Command, Lock, LogOut, Moon, PanelLeftClose, PanelLeft, Sun, User } from "lucide-react";
import { NAV } from "@/lib/nav";
import { api } from "@/lib/api/client";
import { cn } from "@/lib/utils";
import { useTheme } from "@/lib/theme";
import { fetchMe, signOut } from "@/lib/auth";
import { Badge } from "@/components/ui/badge";
import { Tooltip } from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { CommandPalette } from "@/components/ui/command-palette";
import { BrandLockup, LogoMark } from "@/components/ui/logo";

export function AppShell() {
  const { theme, toggle } = useTheme();
  const [collapsed, setCollapsed] = useState(false);
  // A cheap liveness probe that doubles as the header connection indicator.
  const status = useQuery({ queryKey: ["status"], queryFn: api.getStatus, retry: 0 });
  const connected = status.isSuccess;
  const me = useQuery({ queryKey: ["console-me"], queryFn: fetchMe, staleTime: 300_000 });

  return (
    <div className={cn("grid min-h-screen", collapsed ? "grid-cols-[64px_1fr]" : "grid-cols-[248px_1fr]")}>
      <CommandPalette />

      <aside className="flex flex-col border-r bg-sidebar text-sidebar-foreground">
        <div className="flex h-14 items-center gap-2 border-b px-4">
          {collapsed ? <LogoMark size={32} /> : <BrandLockup sublabel="CE Console" size={32} />}
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="ml-auto rounded-md p-1 text-muted-foreground hover:bg-muted"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <PanelLeft size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto scroll-thin px-2.5 py-3">
          {NAV.map((group, gi) => (
            <div key={gi} className="mb-3">
              {group.title && !collapsed && (
                <div className="px-2 pb-1 pt-2 text-2xs font-semibold uppercase tracking-wider text-muted-foreground">
                  {group.title}
                </div>
              )}
              {group.items.map((item) => {
                const link = (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === "/"}
                    className={({ isActive }) =>
                      cn(
                        "flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors",
                        collapsed && "justify-center",
                        isActive
                          ? "bg-primary/10 font-medium text-primary"
                          : "text-foreground/75 hover:bg-muted hover:text-foreground",
                      )
                    }
                  >
                    <item.icon size={16} className="shrink-0" />
                    {!collapsed && <span className="truncate">{item.label}</span>}
                    {!collapsed && item.enterprise && (
                      <Lock size={12} className="ml-auto shrink-0 text-muted-foreground" />
                    )}
                  </NavLink>
                );
                return collapsed ? (
                  <Tooltip key={item.path} content={item.enterprise ? `${item.label} (Enterprise)` : item.label} side="right">
                    {link}
                  </Tooltip>
                ) : (
                  link
                );
              })}
            </div>
          ))}
        </nav>
      </aside>

      <div className="flex min-w-0 flex-col">
        <header className="flex h-14 items-center justify-between gap-3 border-b bg-background px-5">
          <Badge tone="primary">Community Edition</Badge>

          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <button
              onClick={() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))}
              className="hidden items-center gap-1.5 rounded-md border px-2 py-1.5 text-muted-foreground hover:bg-muted sm:flex"
            >
              <Command size={12} /> <span>Search</span>
              <kbd className="rounded border bg-muted px-1 text-2xs">⌘K</kbd>
            </button>
            <Tooltip content={connected ? "Connected to the CE gateway" : "Gateway unreachable"}>
              <Badge tone={connected ? "success" : "warning"}>{connected ? "connected" : "offline"}</Badge>
            </Tooltip>
            <Tooltip content={theme === "dark" ? "Light mode" : "Dark mode"}>
              <button onClick={toggle} className="rounded-md border p-1.5 hover:bg-muted" aria-label="Toggle theme">
                {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
              </button>
            </Tooltip>
            <DropdownMenu>
              <DropdownMenuTrigger className="grid h-8 w-8 place-items-center rounded-full bg-primary/10 text-primary hover:bg-primary/20">
                <User size={15} />
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuLabel>{me.data?.email ?? "Signed in"}</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem danger onSelect={() => { void signOut().then(() => window.location.reload()); }}>
                  <LogOut size={14} /> Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>
        <main className="min-w-0 flex-1 overflow-y-auto scroll-thin p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
