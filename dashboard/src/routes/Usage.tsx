import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, Coins, Cpu, Layers } from "lucide-react";
import { api } from "@/lib/api/client";
import type { UsageEvent } from "@/lib/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/ui/stat-card";
import { EmptyState, ErrorState, SkeletonRows } from "@/components/ui/states";
import { AreaChart, BarChart, DonutChart, Legend, SERIES } from "@/components/charts";
import { formatCompact, formatMicroUsd, formatNumber, formatPercent } from "@/lib/utils";

function costOf(e: UsageEvent): number {
  return e.cost?.canonical_micro_usd ?? 0;
}

export function Usage() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["analytics"],
    queryFn: api.getAnalytics,
  });

  const events = useMemo(() => data ?? [], [data]);

  const totals = useMemo(() => {
    const totalTokens = events.reduce((a, e) => a + e.total_tokens, 0);
    const totalCost = events.reduce((a, e) => a + costOf(e), 0);
    const ok = events.filter((e) => e.success).length;
    return {
      requests: events.length,
      tokens: totalTokens,
      cost: totalCost,
      successRate: events.length > 0 ? ok / events.length : 0,
    };
  }, [events]);

  const byDay = useMemo(() => {
    const map = new Map<string, { date: string; requests: number; tokens: number }>();
    for (const e of events) {
      const date = e.timestamp.slice(0, 10);
      const row = map.get(date) ?? { date, requests: 0, tokens: 0 };
      row.requests += 1;
      row.tokens += e.total_tokens;
      map.set(date, row);
    }
    return [...map.values()].sort((a, b) => a.date.localeCompare(b.date));
  }, [events]);

  const byModel = useMemo(() => {
    const map = new Map<string, { model: string; requests: number; tokens: number; cost: number }>();
    for (const e of events) {
      const row = map.get(e.model) ?? { model: e.model, requests: 0, tokens: 0, cost: 0 };
      row.requests += 1;
      row.tokens += e.total_tokens;
      row.cost += costOf(e);
      map.set(e.model, row);
    }
    return [...map.values()].sort((a, b) => b.requests - a.requests);
  }, [events]);

  const byProvider = useMemo(() => {
    const map = new Map<string, number>();
    for (const e of events) map.set(e.provider, (map.get(e.provider) ?? 0) + 1);
    return [...map.entries()]
      .map(([name, value], i) => ({ name, value, color: SERIES[i % SERIES.length] }))
      .sort((a, b) => b.value - a.value);
  }, [events]);

  const byKey = useMemo(() => {
    const map = new Map<string, { key: string; requests: number; tokens: number; cost: number }>();
    for (const e of events) {
      const k = e.virtual_key_name || "—";
      const row = map.get(k) ?? { key: k, requests: 0, tokens: 0, cost: 0 };
      row.requests += 1;
      row.tokens += e.total_tokens;
      row.cost += costOf(e);
      map.set(k, row);
    }
    return [...map.values()].sort((a, b) => b.requests - a.requests);
  }, [events]);

  return (
    <>
      <PageHeader
        title="Usage & Analytics"
        description="Aggregated from the gateway's recent usage events (the in-memory observability ring). Grouped by model, provider, and key."
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={Activity} label="Requests" value={isLoading ? "…" : formatNumber(totals.requests)} loading={isLoading} sub={`${formatPercent(totals.successRate)} success`} />
        <StatCard icon={Cpu} label="Tokens" value={isLoading ? "…" : formatCompact(totals.tokens)} loading={isLoading} sub="prompt + completion" />
        <StatCard icon={Coins} label="Cost" value={isLoading ? "…" : formatMicroUsd(totals.cost)} loading={isLoading} sub="canonical USD" />
        <StatCard icon={Layers} label="Models used" value={isLoading ? "…" : formatNumber(byModel.length)} loading={isLoading} sub={`${byProvider.length} provider${byProvider.length === 1 ? "" : "s"}`} />
      </div>

      {isError ? (
        <Card className="mt-4">
          <CardBody>
            <ErrorState message="The gateway's /analytics didn't respond." onRetry={() => refetch()} />
          </CardBody>
        </Card>
      ) : isLoading ? (
        <Card className="mt-4">
          <CardBody>
            <SkeletonRows rows={8} />
          </CardBody>
        </Card>
      ) : events.length === 0 ? (
        <Card className="mt-4">
          <CardBody>
            <EmptyState
              icon={Activity}
              title="No usage yet"
              description="This is expected on a fresh install. Send requests through the gateway and analytics will populate here."
            />
          </CardBody>
        </Card>
      ) : (
        <>
          <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader title="Requests over time" description="Grouped by day." />
              <CardBody>
                <AreaChart
                  data={byDay}
                  xKey="date"
                  series={[{ key: "requests", name: "Requests", color: "hsl(var(--chart-1))" }]}
                  yFormatter={(v) => formatCompact(v)}
                  height={240}
                />
              </CardBody>
            </Card>

            <Card>
              <CardHeader title="By provider" description="Share of requests." />
              <CardBody className="space-y-4">
                <DonutChart data={byProvider} valueFormatter={(v) => formatNumber(v)} />
                <Legend
                  items={byProvider.map((p) => ({ name: p.name, value: formatNumber(p.value), color: p.color }))}
                />
              </CardBody>
            </Card>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader title="Tokens by model" />
              <CardBody>
                <BarChart
                  data={byModel.slice(0, 8)}
                  xKey="model"
                  series={[{ key: "tokens", name: "Tokens", color: "hsl(var(--chart-2))" }]}
                  yFormatter={(v) => formatCompact(v)}
                  height={260}
                />
              </CardBody>
            </Card>

            <Card>
              <CardHeader title="By virtual key" description="Requests, tokens, and cost per key." />
              <CardBody className="p-0">
                <div className="overflow-x-auto scroll-thin">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-xs text-muted-foreground">
                        <th className="px-5 py-2.5 text-left font-medium">Key</th>
                        <th className="px-5 py-2.5 text-right font-medium">Requests</th>
                        <th className="px-5 py-2.5 text-right font-medium">Tokens</th>
                        <th className="px-5 py-2.5 text-right font-medium">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {byKey.map((k) => (
                        <tr key={k.key} className="border-b last:border-0">
                          <td className="px-5 py-2.5 font-mono text-xs">{k.key}</td>
                          <td className="px-5 py-2.5 text-right tnum">{formatNumber(k.requests)}</td>
                          <td className="px-5 py-2.5 text-right tnum text-muted-foreground">{formatCompact(k.tokens)}</td>
                          <td className="px-5 py-2.5 text-right tnum">{formatMicroUsd(k.cost)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardBody>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
