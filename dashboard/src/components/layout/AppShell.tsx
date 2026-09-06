import { useEffect, useRef, useState } from "react";
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
import { BrandLockup } from "@/components/ui/logo";
import { Drawer, DrawerContent, DrawerHeader, DrawerTrigger } from "@/components/ui/drawer";

export function AppShell() {
  const { theme, toggle } = useTheme();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const mobileTrigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const media = window.matchMedia("(min-width: 768px)");
    const closeOnDesktop = () => { if (media.matches) setMobileOpen(false); };
    media.addEventListener("change", closeOnDesktop);
    return () => media.removeEventListener("change", closeOnDesktop);
  }, []);
  // A cheap liveness probe that doubles as the header connection indicator.
  const status = useQuery({ queryKey: ["status"], queryFn: api.getStatus, retry: 0 });
  const connected = status.isSuccess;
  const me = useQuery({ queryKey: ["console-me"], queryFn: fetchMe, staleTime: 300_000 });

  return (
    <div className={cn("grid min-h-screen grid-cols-[minmax(0,1fr)]", collapsed ? "md:grid-cols-[64px_minmax(0,1fr)]" : "md:grid-cols-[248px_minmax(0,1fr)]")}>
      <CommandPalette />

      <aside className="hidden min-w-0 flex-col border-r bg-sidebar text-sidebar-foreground md:flex">
        <div className={cn("flex h-14 items-center border-b", collapsed ? "justify-center px-2" : "gap-2 px-4")}>
          {!collapsed && <BrandLockup sublabel="CE Console" size={32} />}
          <button
            onClick={() => setCollapsed((c) => !c)}
            className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-md text-muted-foreground hover:bg-muted", !collapsed && "ml-auto")}
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
                    aria-label={item.label}
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
        <header className="flex min-h-14 flex-wrap items-center justify-between gap-2 border-b bg-background px-3 py-2 sm:px-5">
          <div className="flex min-w-0 items-center gap-2">
            <Drawer open={mobileOpen} onOpenChange={setMobileOpen}>
              <DrawerTrigger asChild>
                <button ref={mobileTrigger} aria-label="Open navigation" className="rounded-md border p-2 md:hidden"><PanelLeft size={16} /></button>
              </DrawerTrigger>
              <DrawerContent className="left-0 right-auto w-[min(20rem,calc(100vw-2rem))]" onCloseAutoFocus={(event) => {
                event.preventDefault();
                // On resize the mobile trigger is hidden; focus the desktop navigation instead.
                if (window.matchMedia("(min-width: 768px)").matches) document.querySelector<HTMLAnchorElement>("aside nav a")?.focus();
                else mobileTrigger.current?.focus();
              }}>
                <DrawerHeader title="Navigation" description="Community Edition Console" />
                <nav aria-label="Mobile navigation" className="min-h-0 overflow-y-auto p-3">
                  {NAV.map((group, gi) => <div key={gi} className="mb-3">
                    {group.title && <div className="px-2 py-2 text-xs text-muted-foreground">{group.title}</div>}
                    {group.items.map((item) => <NavLink key={item.path} to={item.path} end={item.path === "/"} onClick={() => setMobileOpen(false)} className={({ isActive }) => cn("flex items-center gap-2 rounded-md p-2 text-sm", isActive ? "bg-primary/10 text-primary" : "hover:bg-muted")}>
                      <item.icon size={16} />{item.label}{item.enterprise && <Lock size={12} className="ml-auto" />}
                    </NavLink>)}
                  </div>)}
                </nav>
              </DrawerContent>
            </Drawer>
            <Badge tone="primary">Community Edition</Badge>
          </div>

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
              <DropdownMenuTrigger aria-label="Account menu" className="grid h-8 w-8 place-items-center rounded-full bg-primary/10 text-primary hover:bg-primary/20">
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
        <main className="min-w-0 flex-1 scroll-thin p-3 sm:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
