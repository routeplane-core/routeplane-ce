import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogClose = DialogPrimitive.Close;

export function DialogContent({
  className,
  children,
  size = "md",
}: {
  className?: string;
  children: ReactNode;
  size?: "sm" | "md" | "lg";
}) {
  const widths = { sm: "max-w-md", md: "max-w-lg", lg: "max-w-2xl" };
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm animate-overlay-in" />
      <DialogPrimitive.Content
        className={cn(
          "fixed left-1/2 top-1/2 z-50 w-[calc(100vw-2rem)] -translate-x-1/2 -translate-y-1/2 rounded-lg border bg-card shadow-xl animate-fade-in",
          widths[size],
          className,
        )}
      >
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}

export function DialogHeader({ title, description }: { title: ReactNode; description?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b px-5 py-4">
      <div>
        <DialogPrimitive.Title className="text-sm font-semibold tracking-tight">{title}</DialogPrimitive.Title>
        {description && (
          <DialogPrimitive.Description className="mt-0.5 text-sm text-muted-foreground">
            {description}
          </DialogPrimitive.Description>
        )}
      </div>
      <DialogPrimitive.Close className="rounded-md p-1 text-muted-foreground hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <X size={16} />
      </DialogPrimitive.Close>
    </div>
  );
}

export function DialogBody({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("max-h-[70vh] overflow-y-auto scroll-thin px-5 py-4", className)}>{children}</div>;
}

export function DialogFooter({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("flex items-center justify-end gap-2 border-t px-5 py-3", className)}>{children}</div>;
}
