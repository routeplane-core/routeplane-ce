import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

// A right-side sheet for detail drill-downs (logs, traces, agent runs).
export const Drawer = DialogPrimitive.Root;
export const DrawerTrigger = DialogPrimitive.Trigger;
export const DrawerClose = DialogPrimitive.Close;

export function DrawerContent({
  className,
  children,
  size = "md",
}: {
  className?: string;
  children: ReactNode;
  size?: "md" | "lg" | "xl";
}) {
  const widths = { md: "max-w-md", lg: "max-w-xl", xl: "max-w-3xl" };
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm animate-overlay-in" />
      <DialogPrimitive.Content
        className={cn(
          "fixed inset-y-0 right-0 z-50 flex w-[calc(100vw-3rem)] flex-col border-l bg-card shadow-xl outline-none",
          "data-[state=open]:animate-fade-in",
          widths[size],
          className,
        )}
      >
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}

export function DrawerHeader({
  title,
  description,
  action,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3 border-b px-5 py-4">
      <div className="min-w-0">
        <DialogPrimitive.Title className="truncate text-sm font-semibold tracking-tight">{title}</DialogPrimitive.Title>
        {description && (
          <DialogPrimitive.Description className="mt-0.5 text-sm text-muted-foreground">
            {description}
          </DialogPrimitive.Description>
        )}
      </div>
      <div className="flex items-center gap-1">
        {action}
        <DialogPrimitive.Close className="rounded-md p-1 text-muted-foreground hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
          <X size={16} />
        </DialogPrimitive.Close>
      </div>
    </div>
  );
}

export function DrawerBody({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("flex-1 overflow-y-auto scroll-thin px-5 py-4", className)}>{children}</div>;
}
