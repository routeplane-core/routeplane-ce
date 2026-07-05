import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Activity, Boxes, CheckCircle2, DatabaseZap, HeartPulse } from "lucide-react";
import { api } from "@/lib/api/client";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/ui/stat-card";
import { EmptyState, ErrorState, SkeletonRows } from "@/components/ui/states";
import { BarChart } from "@/components/charts";
import { formatCompact, formatNumber, formatPercent, formatRelativeTime } from "@/lib/utils";

const OUTCOME_TONE = (ok: boolean) => (ok ? "success" : "danger");

export function Overview() {
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: api.getMetrics });
  const status = useQuery({ queryKey: ["status"], queryFn: api.getStatus });
  const models = useQuery({ queryKey: ["models"], queryFn: api.getModels });
  const analytics = useQuery({ queryKey: ["analytics"], queryFn: api.getAnalytics });

  const m = metrics.data ?? [];
  const totalRequests = m.reduce((a, x) => a + x.value, 0);
  const successRequests = m
    .filter((x) => x.outcome === "success")
    .reduce((a, x) => a + x.value, 0);
  const successRate = totalRequests > 0 ? successRequests / totalRequests : 0;

  const cache = status.data?.cache;
  const providers = status.data?.providers ?? [];
  const healthy = providers.filter((p) => p.circuit === "closed").length;
  const openCircuits = providers.filter((p) => p.circuit === "open").length;

  const modelCount = models.data?.data.length ?? 0;

  const byProvider = (() => {
    const map = new Map<string, number>();
    for (const x of m) map.set(x.provider, (map.get(x.provider) ?? 0) + x.value);
    return [...map.entries()]
      .map(([provider, requests]) => ({ provider, requests }))
      .sort((a, b) => b.requests - a.requests);
  })();

  const events = [...(analytics.data ?? [])]
    .sort((a, b) => b.timestamp.localeCompare(a.timestamp))
    .slice(0, 8);

  return (
    <>
      <PageHeader
        title="Overview"
        description="Your gateway at a glance — traffic, cache efficiency, catalog size, and provider health, read live from the CE data plane."
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={Activity}
          label="Total requests"
          value={metrics.isLoading ? "…" : formatNumber(totalRequests)}
          loading={metrics.isLoading}
          sub={metrics.isError ? "metrics unavailable" : `${formatPercent(successRate)} success`}
        />
        <StatCard
          icon={CheckCircle2}
          label="Success rate"
          value={metrics.isLoading ? "…" : totalRequests > 0 ? formatPercent(successRate) : "—"}
          loading={metrics.isLoading}
          sub={`${formatNumber(successRequests)} succeeded`}
        />
        <StatCard
          icon={DatabaseZap}
          label="Cache hit-rate"
          value={status.isLoading ? "…" : cache ? formatPercent(cache.hit_rate) : "—"}
          loading={status.isLoading}
          sub={cache ? `${formatCompact(cache.entries)} entries` : "exact-match cache"}
        />
        <StatCard
          icon={Boxes}
          label="Models"
          value={models.isLoading ? "…" : formatNumber(modelCount)}
          loading={models.isLoading}
          sub={`${providers.length} provider${providers.length === 1 ? "" : "s"} wired`}
        />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Requests by provider"
            description="Counters from the gateway's Prometheus /metrics surface."
            action={
              <Link to="/health" className="text-xs text-primary hover:underline">
                Health →
              </Link>
            }
          />
          <CardBody>
            {metrics.isLoading ? (
              <SkeletonRows rows={5} />
            ) : metrics.isError ? (
              <ErrorState message="The gateway didn't return metrics." onRetry={() => metrics.refetch()} />
            ) : byProvider.length === 0 ? (
              <EmptyState
                icon={Activity}
                title="No traffic yet"
                description="Send a request through the gateway and it will appear here."
              />
            ) : (
              <BarChart
                data={byProvider}
                xKey="provider"
                series={[{ key: "requests", name: "Requests", color: "hsl(var(--chart-1))" }]}
                yFormatter={(v) => formatCompact(v)}
                height={240}
              />
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Provider health"
            action={
              <Link to="/health" className="text-xs text-primary hover:underline">
                Details →
              </Link>
            }
          />
          <CardBody className="space-y-3">
            {status.isLoading ? (
              <SkeletonRows rows={3} />
            ) : status.isError ? (
              <ErrorState message="The gateway's /status didn't respond." onRetry={() => status.refetch()} />
            ) : providers.length === 0 ? (
              <div className="py-6 text-center text-sm text-muted-foreground">No providers reporting.</div>
            ) : (
              <>
                <div className="flex items-center justify-between rounded-md border px-3 py-2 text-sm">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <HeartPulse size={14} /> Healthy circuits
                  </span>
                  <Badge tone="success">{healthy}</Badge>
                </div>
                <div className="flex items-center justify-between rounded-md border px-3 py-2 text-sm">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <HeartPulse size={14} /> Open circuits
                  </span>
                  <Badge tone={openCircuits > 0 ? "danger" : "neutral"}>{openCircuits}</Badge>
                </div>
                <ul className="space-y-1.5 pt-1">
                  {providers.slice(0, 6).map((p) => (
                    <li key={p.provider} className="flex items-center justify-between gap-3 text-xs">
                      <span className="truncate font-mono">{p.provider}</span>
                      <Badge
                        tone={p.circuit === "closed" ? "success" : p.circuit === "open" ? "danger" : "warning"}
                      >
                        {p.circuit}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </CardBody>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader
          title="Recent activity"
          description="The latest usage events from the in-memory observability ring."
          action={
            <Link to="/logs" className="text-xs text-primary hover:underline">
              All logs →
            </Link>
          }
        />
        <CardBody className="p-0">
          {analytics.isLoading ? (
            <div className="p-5">
              <SkeletonRows rows={6} />
            </div>
          ) : analytics.isError ? (
            <ErrorState message="The gateway's /analytics didn't respond." onRetry={() => analytics.refetch()} />
          ) : events.length === 0 ? (
            <EmptyState
              icon={Activity}
              title="No activity yet"
              description="Once requests flow through the gateway, recent events show up here."
            />
          ) : (
            <ul className="divide-y">
              {events.map((e, i) => (
                <li key={`${e.timestamp}-${i}`} className="flex items-center gap-3 px-5 py-2.5 text-sm">
                  <Badge tone={OUTCOME_TONE(e.success)}>{e.success ? "ok" : "error"}</Badge>
                  <span className="truncate font-mono text-xs">{e.model}</span>
                  <span className="truncate text-xs text-muted-foreground">{e.provider}</span>
                  <span className="ml-auto shrink-0 text-xs text-muted-foreground tnum">
                    {formatCompact(e.total_tokens)} tok
                  </span>
                  <span className="shrink-0 text-xs text-muted-foreground">{formatRelativeTime(e.timestamp)}</span>
                </li>
              ))}
            </ul>
          )}
        </CardBody>
      </Card>
    </>
  );
}
