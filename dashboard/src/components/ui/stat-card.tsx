import type { ReactNode } from "react";
import { ArrowDownRight, ArrowUpRight, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardBody } from "./card";
import { Sparkline } from "@/components/charts";

export interface StatCardProps {
  icon?: LucideIcon;
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  /** Period-over-period change, as a ratio (e.g. +0.12 = +12%). */
  delta?: number;
  /** When true, a downward delta is "good" (e.g. error rate, latency). */
  invertDelta?: boolean;
  spark?: { v: number }[];
  sparkColor?: string;
  loading?: boolean;
}

export function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  delta,
  invertDelta,
  spark,
  sparkColor,
  loading,
}: StatCardProps) {
  const positive = delta != null && (invertDelta ? delta < 0 : delta > 0);
  const negative = delta != null && (invertDelta ? delta > 0 : delta < 0);
  return (
    <Card>
      <CardBody className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            {Icon && <Icon size={14} />}
            {label}
          </div>
          {delta != null && (
            <span
              className={cn(
                "inline-flex items-center gap-0.5 text-2xs font-medium tnum",
                positive && "text-success",
                negative && "text-danger",
                !positive && !negative && "text-muted-foreground",
              )}
            >
              {delta >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
              {Math.abs(delta * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <div className="mt-2 flex items-end justify-between gap-3">
          <div>
            <div className={cn("text-2xl font-semibold tracking-tight tnum", loading && "animate-pulse text-muted-foreground")}>
              {value}
            </div>
            {sub && <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>}
          </div>
          {spark && spark.length > 1 && (
            <div className="h-9 w-24 shrink-0">
              <Sparkline data={spark} color={sparkColor ?? "hsl(var(--primary))"} />
            </div>
          )}
        </div>
      </CardBody>
    </Card>
  );
}
