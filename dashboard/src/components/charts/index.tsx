import {
  Area,
  AreaChart as RAreaChart,
  Bar,
  BarChart as RBarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart as RLineChart,
  Pie,
  PieChart as RPieChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReactNode } from "react";

// Themed Recharts wrappers — bound to our CSS-var palette so charts never show
// the default Recharts blue/green and read as part of the design system.
export const SERIES = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
  "hsl(var(--chart-6))",
];

const axis = {
  stroke: "hsl(var(--border))",
  tick: { fill: "hsl(var(--muted-foreground))", fontSize: 11 },
  tickLine: false,
  axisLine: false,
};

function ChartTooltip({ active, payload, label, formatter }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border bg-popover px-2.5 py-1.5 text-xs shadow-md">
      {label != null && <div className="mb-1 font-medium text-foreground">{label}</div>}
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2 tnum">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color || p.fill }} />
          <span className="text-muted-foreground">{p.name}</span>
          <span className="ml-auto font-medium text-foreground">
            {formatter ? formatter(p.value, p.name) : p.value?.toLocaleString?.() ?? p.value}
          </span>
        </div>
      ))}
    </div>
  );
}

type SeriesDef = { key: string; name: string; color?: string };

interface BaseProps {
  data: any[];
  xKey: string;
  series: SeriesDef[];
  height?: number;
  yFormatter?: (v: number) => string;
  tipFormatter?: (v: number, name: string) => ReactNode;
  hideXAxis?: boolean;
}

export function AreaChart({ data, xKey, series, height = 240, yFormatter, tipFormatter, hideXAxis }: BaseProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RAreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <defs>
          {series.map((s, i) => (
            <linearGradient key={s.key} id={`g-${s.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color ?? SERIES[i % SERIES.length]} stopOpacity={0.25} />
              <stop offset="100%" stopColor={s.color ?? SERIES[i % SERIES.length]} stopOpacity={0.02} />
            </linearGradient>
          ))}
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        {!hideXAxis && <XAxis dataKey={xKey} {...axis} minTickGap={24} />}
        <YAxis {...axis} width={48} tickFormatter={yFormatter} />
        <RTooltip content={<ChartTooltip formatter={tipFormatter} />} />
        {series.map((s, i) => (
          <Area
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.name}
            stroke={s.color ?? SERIES[i % SERIES.length]}
            strokeWidth={2}
            fill={`url(#g-${s.key})`}
            isAnimationActive={false}
          />
        ))}
      </RAreaChart>
    </ResponsiveContainer>
  );
}

export function LineChart({ data, xKey, series, height = 240, yFormatter, tipFormatter }: BaseProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RLineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis dataKey={xKey} {...axis} minTickGap={24} />
        <YAxis {...axis} width={48} tickFormatter={yFormatter} />
        <RTooltip content={<ChartTooltip formatter={tipFormatter} />} />
        {series.map((s, i) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.name}
            stroke={s.color ?? SERIES[i % SERIES.length]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </RLineChart>
    </ResponsiveContainer>
  );
}

export function BarChart({
  data,
  xKey,
  series,
  height = 240,
  yFormatter,
  tipFormatter,
  stacked,
}: BaseProps & { stacked?: boolean }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RBarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis dataKey={xKey} {...axis} minTickGap={12} />
        <YAxis {...axis} width={48} tickFormatter={yFormatter} />
        <RTooltip cursor={{ fill: "hsl(var(--muted))", opacity: 0.4 }} content={<ChartTooltip formatter={tipFormatter} />} />
        {series.map((s, i) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            name={s.name}
            stackId={stacked ? "a" : undefined}
            fill={s.color ?? SERIES[i % SERIES.length]}
            radius={stacked ? 0 : [3, 3, 0, 0]}
            maxBarSize={48}
            isAnimationActive={false}
          />
        ))}
      </RBarChart>
    </ResponsiveContainer>
  );
}

export function DonutChart({
  data,
  height = 220,
  valueFormatter,
}: {
  data: { name: string; value: number; color?: string }[];
  height?: number;
  valueFormatter?: (v: number) => ReactNode;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RPieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="82%" paddingAngle={2} stroke="none" isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color ?? SERIES[i % SERIES.length]} />
          ))}
        </Pie>
        <RTooltip content={<ChartTooltip formatter={valueFormatter} />} />
      </RPieChart>
    </ResponsiveContainer>
  );
}

/** Inline sparkline for stat cards / table rows. */
export function Sparkline({
  data,
  dataKey = "v",
  color = "hsl(var(--primary))",
  height = 36,
}: {
  data: any[];
  dataKey?: string;
  color?: string;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RAreaChart data={data} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id={`sp-${dataKey}-${color}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5} fill={`url(#sp-${dataKey}-${color})`} isAnimationActive={false} />
      </RAreaChart>
    </ResponsiveContainer>
  );
}

/** Discrete legend, paired with DonutChart. */
export function Legend({ items }: { items: { name: string; value?: ReactNode; color: string }[] }) {
  return (
    <ul className="space-y-1.5 text-sm">
      {items.map((it) => (
        <li key={it.name} className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: it.color }} />
          <span className="truncate text-muted-foreground">{it.name}</span>
          {it.value != null && <span className="ml-auto font-medium tnum">{it.value}</span>}
        </li>
      ))}
    </ul>
  );
}
