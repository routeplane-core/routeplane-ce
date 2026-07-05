import { useEffect, useState } from "react";
import { Command } from "cmdk";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { NAV } from "@/lib/nav";

// Global ⌘K / Ctrl-K navigator + search over the full IA.
export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const go = (path: string) => {
    navigate(path);
    setOpen(false);
  };

  return (
    <Command.Dialog
      open={open}
      onOpenChange={setOpen}
      label="Command palette"
      className="fixed left-1/2 top-[18%] z-[110] w-[calc(100vw-2rem)] max-w-lg -translate-x-1/2 overflow-hidden rounded-lg border bg-popover shadow-xl animate-fade-in"
    >
      {open && <div className="fixed inset-0 -z-10 bg-black/40 backdrop-blur-sm" onClick={() => setOpen(false)} />}
      <div className="flex items-center gap-2 border-b px-3">
        <Search size={16} className="text-muted-foreground" />
        <Command.Input
          placeholder="Jump to…"
          className="h-11 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
        />
        <kbd className="rounded border bg-muted px-1.5 py-0.5 text-2xs text-muted-foreground">ESC</kbd>
      </div>
      <Command.List className="max-h-80 overflow-y-auto scroll-thin p-2">
        <Command.Empty className="py-8 text-center text-sm text-muted-foreground">No matches.</Command.Empty>
        {NAV.map((group, gi) => (
          <Command.Group
            key={gi}
            heading={group.title || "General"}
            className="px-1 py-1 text-2xs font-semibold uppercase tracking-wider text-muted-foreground [&_[cmdk-group-heading]]:px-1 [&_[cmdk-group-heading]]:pb-1"
          >
            {group.items.map((item) => (
              <Command.Item
                key={item.path}
                value={`${item.label} ${item.path}`}
                onSelect={() => go(item.path)}
                className="flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-2 text-sm text-foreground aria-selected:bg-muted"
              >
                <item.icon size={15} className="text-muted-foreground" />
                {item.label}
              </Command.Item>
            ))}
          </Command.Group>
        ))}
      </Command.List>
    </Command.Dialog>
  );
}
