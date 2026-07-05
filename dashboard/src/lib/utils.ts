import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes with conflict resolution (shadcn/ui convention). */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format integer micro-USD as a currency string (default USD). */
export function formatMicroUsd(microUsd: number, currency = "USD"): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: microUsd < 1_000_000 ? 4 : 2,
  }).format(microUsd / 1_000_000);
}

/** Format a value already expressed in a currency's major units. */
export function formatCurrency(amount: number, currency = "USD"): string {
  return new Intl.NumberFormat(currency === "INR" ? "en-IN" : "en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(amount);
}

/** Compact number formatting (e.g. 12.3k). */
export function formatCompact(n: number): string {
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

/** Full grouped integer (e.g. 1,234,567). */
export function formatNumber(n: number): string {
  return new Intl.NumberFormat("en-US").format(Math.round(n));
}

/** Tokens, compacted with a unit suffix. */
export function formatTokens(n: number): string {
  return `${formatCompact(n)} tok`;
}

/** Ratio (0..1) → percent string with one decimal. */
export function formatPercent(ratio: number, digits = 1): string {
  return `${(ratio * 100).toFixed(digits)}%`;
}

/** Milliseconds → human latency (e.g. 842ms, 1.24s). */
export function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

const RELATIVE = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
const UNITS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["year", 31_536_000_000],
  ["month", 2_592_000_000],
  ["day", 86_400_000],
  ["hour", 3_600_000],
  ["minute", 60_000],
  ["second", 1000],
];

/** ISO string → relative time (e.g. "3 hours ago"). Pure (uses passed-in now). */
export function formatRelativeTime(iso: string, now: number = Date.now()): string {
  const diff = new Date(iso).getTime() - now;
  for (const [unit, ms] of UNITS) {
    if (Math.abs(diff) >= ms || unit === "second") {
      return RELATIVE.format(Math.round(diff / ms), unit);
    }
  }
  return "just now";
}

/** ISO string → "Jun 12, 14:32". */
export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

/** ISO string → "2026-06-12". */
export function formatDate(iso: string): string {
  return iso.slice(0, 10);
}

/** Currency display metadata for the multi-currency surfaces. */
export const CURRENCIES: { code: string; symbol: string; label: string }[] = [
  { code: "USD", symbol: "$", label: "US Dollar" },
  { code: "INR", symbol: "₹", label: "Indian Rupee" },
  { code: "EUR", symbol: "€", label: "Euro" },
  { code: "AED", symbol: "د.إ", label: "UAE Dirham" },
];

/** Indicative FX from USD — mirrors the configurable FX layer (PRD-015). */
export const FX_FROM_USD: Record<string, number> = { USD: 1, INR: 83.4, EUR: 0.92, AED: 3.67 };

/** Convert a micro-USD amount to a target currency's major units. */
export function convertMicroUsd(microUsd: number, currency: string): number {
  return (microUsd / 1_000_000) * (FX_FROM_USD[currency] ?? 1);
}

/** Trigger a client-side download of text content (CSV/JSON exports). */
export function downloadText(filename: string, content: string, mime = "text/plain") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
