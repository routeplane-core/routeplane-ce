import * as DM from "@radix-ui/react-dropdown-menu";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export const DropdownMenu = DM.Root;
export const DropdownMenuTrigger = DM.Trigger;
export const DropdownMenuSeparator = () => <DM.Separator className="my-1 h-px bg-border" />;

export function DropdownMenuContent({
  className,
  children,
  align = "end",
}: {
  className?: string;
  children: ReactNode;
  align?: "start" | "center" | "end";
}) {
  return (
    <DM.Portal>
      <DM.Content
        align={align}
        sideOffset={6}
        className={cn(
          "z-50 min-w-44 overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-lg animate-fade-in",
          className,
        )}
      >
        {children}
      </DM.Content>
    </DM.Portal>
  );
}

export function DropdownMenuItem({
  className,
  children,
  onSelect,
  danger,
  disabled,
}: {
  className?: string;
  children: ReactNode;
  onSelect?: () => void;
  danger?: boolean;
  disabled?: boolean;
}) {
  return (
    <DM.Item
      disabled={disabled}
      onSelect={onSelect}
      className={cn(
        "flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none data-[highlighted]:bg-muted data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
        danger && "text-danger data-[highlighted]:bg-danger/10",
        className,
      )}
    >
      {children}
    </DM.Item>
  );
}

export function DropdownMenuLabel({ children }: { children: ReactNode }) {
  return <DM.Label className="px-2 py-1.5 text-2xs font-semibold uppercase tracking-wider text-muted-foreground">{children}</DM.Label>;
}
