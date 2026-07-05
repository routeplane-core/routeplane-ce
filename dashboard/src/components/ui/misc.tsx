import { useState, type ReactNode } from "react";
import { Check, ChevronLeft, ChevronRight, Copy } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./button";

/** A horizontal determinate progress bar. tone drives the fill color. */
export function Progress({
  value,
  tone = "primary",
  className,
}: {
  value: number; // 0..1
  tone?: "primary" | "success" | "warning" | "danger";
  className?: string;
}) {
  const fills = {
    primary: "bg-primary",
    success: "bg-success",
    warning: "bg-warning",
    danger: "bg-danger",
  };
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-muted", className)}>
      <div
        className={cn("h-full rounded-full transition-all", fills[tone])}
        style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
      />
    </div>
  );
}

/** Segmented control — a compact set of mutually-exclusive options. */
export function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  className,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode }[];
  className?: string;
}) {
  return (
    <div className={cn("inline-flex rounded-md border bg-muted/50 p-0.5", className)}>
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-[5px] px-2.5 py-1 text-xs font-medium transition-colors",
            value === o.value ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Pagination({
  page,
  pageCount,
  onPage,
  total,
}: {
  page: number;
  pageCount: number;
  onPage: (p: number) => void;
  total?: number;
}) {
  return (
    <div className="flex items-center justify-between px-5 py-3 text-xs text-muted-foreground">
      <span>
        Page {page + 1} of {pageCount}
        {total != null && ` · ${total.toLocaleString()} rows`}
      </span>
      <div className="flex items-center gap-1">
        <Button variant="outline" size="icon" className="h-7 w-7" disabled={page <= 0} onClick={() => onPage(page - 1)}>
          <ChevronLeft size={14} />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="h-7 w-7"
          disabled={page >= pageCount - 1}
          onClick={() => onPage(page + 1)}
        >
          <ChevronRight size={14} />
        </Button>
      </div>
    </div>
  );
}

export function CopyButton({ value, label, className }: { value: string; label?: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="outline"
      size="sm"
      className={className}
      onClick={() => {
        navigator.clipboard?.writeText(value);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
      {label ?? (copied ? "Copied" : "Copy")}
    </Button>
  );
}

export function CodeBlock({ code, lang, className }: { code: string; lang?: string; className?: string }) {
  return (
    <div className={cn("group relative overflow-hidden rounded-md border bg-muted/40", className)}>
      {lang && (
        <div className="border-b px-3 py-1.5 text-2xs font-medium uppercase tracking-wider text-muted-foreground">
          {lang}
        </div>
      )}
      <pre className="overflow-x-auto scroll-thin px-3 py-2.5 text-xs leading-relaxed">
        <code className="font-mono">{code}</code>
      </pre>
      <div className="absolute right-2 top-2 opacity-0 transition-opacity group-hover:opacity-100">
        <CopyButton value={code} label="" className="h-7 w-7 px-0" />
      </div>
    </div>
  );
}

/** A definition list for detail panels — label/value rows. */
export function KeyValueList({
  items,
  className,
}: {
  items: { label: ReactNode; value: ReactNode }[];
  className?: string;
}) {
  return (
    <dl className={cn("divide-y text-sm", className)}>
      {items.map((it, i) => (
        <div key={i} className="flex items-start justify-between gap-4 py-2">
          <dt className="shrink-0 text-muted-foreground">{it.label}</dt>
          <dd className="min-w-0 break-words text-right font-medium">{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}
