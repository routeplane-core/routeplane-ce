import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { HeartPulse } from "lucide-react";
import { api } from "@/lib/api/client";
import type { ProviderStatus } from "@/lib/api/types";
import { Card, CardBody } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/layout/PageHeader";
import { DataTable, type Column } from "@/components/ui/table";
import { SkeletonRows, EmptyState, ErrorState } from "@/components/ui/states";
import { formatMs, formatNumber, formatPercent } from "@/lib/utils";

interface HealthRow {
  provider: string;
  circuit: ProviderStatus["circuit"];
  latencyEwmaMs: number | null;
  requests: number;
  errors: number;
}

const CIRCUIT_TONE = (c: string) =>
  c === "closed" ? "success" : c === "open" ? "danger" : c === "half_open" ? "warning" : "neutral";

const CIRCUIT_LABEL = (c: string) => (c === "half_open" ? "half-open" : c);

export function Health() {
  const status = useQuery({ queryKey: ["status"], queryFn: api.getStatus });
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: api.getMetrics });

  const rows = useMemo<HealthRow[]>(() => {
    const providers = status.data?.providers ?? [];
    const m = metrics.data ?? [];

    const circuitByName = new Map(providers.map((p) => [p.provider, p]));
    const reqByName = new Map<string, number>();
    const errByName = new Map<string, number>();
    for (const x of m) {
      reqByName.set(x.provider, (reqByName.get(x.provider) ?? 0) + x.value);
      if (x.outcome !== "success") errByName.set(x.provider, (errByName.get(x.provider) ?? 0) + x.value);
    }

    const names = new Set<string>([...circuitByName.keys(), ...reqByName.keys()]);
    return [...names]
      .map((name) => {
        const p = circuitByName.get(name);
        return {
          provider: name,
          circuit: p?.circuit ?? "closed",
          latencyEwmaMs: p?.latency_ewma_ms ?? null,
          requests: reqByName.get(name) ?? 0,
          errors: errByName.get(name) ?? 0,
        };
      })
      .sort((a, b) => b.requests - a.requests);
  }, [status.data, metrics.data]);

  const columns: Column<HealthRow>[] = [
    { key: "provider", header: "Provider", cell: (r) => <span className="font-mono text-xs">{r.provider}</span> },
    {
      key: "circuit",
      header: "Circuit",
      cell: (r) => <Badge tone={CIRCUIT_TONE(r.circuit)}>{CIRCUIT_LABEL(r.circuit)}</Badge>,
    },
    {
      key: "latency",
      header: "EWMA latency",
      align: "right",
      sortValue: (r) => r.latencyEwmaMs ?? -1,
      cell: (r) => <span className="tnum">{r.latencyEwmaMs != null ? formatMs(r.latencyEwmaMs) : "—"}</span>,
    },
    {
      key: "requests",
      header: "Requests",
      align: "right",
      sortValue: (r) => r.requests,
      cell: (r) => <span className="tnum text-muted-foreground">{formatNumber(r.requests)}</span>,
    },
    {
      key: "errors",
      header: "Errors",
      align: "right",
      sortValue: (r) => r.errors,
      cell: (r) => <span className="tnum">{formatNumber(r.errors)}</span>,
    },
    {
      key: "errorRate",
      header: "Error rate",
      align: "right",
      sortValue: (r) => (r.requests > 0 ? r.errors / r.requests : 0),
      cell: (r) => (
        <span className="tnum">{r.requests > 0 ? formatPercent(r.errors / r.requests) : "—"}</span>
      ),
    },
  ];

  const isLoading = status.isLoading || metrics.isLoading;
  const isError = status.isError;

  return (
    <>
      <PageHeader
        title="Provider Health"
        description="Circuit-breaker state and EWMA latency per provider from /status, joined with request and error counters from /metrics."
      />

      <Card>
        <CardBody className="p-0">
          {isLoading ? (
            <div className="p-5">
              <SkeletonRows rows={6} />
            </div>
          ) : isError ? (
            <ErrorState message="The gateway's /status didn't respond." onRetry={() => status.refetch()} />
          ) : rows.length === 0 ? (
            <EmptyState
              icon={HeartPulse}
              title="No providers reporting"
              description="Once a provider is configured and receives traffic, its circuit state and latency show up here."
            />
          ) : (
            <DataTable columns={columns} rows={rows} getRowId={(r) => r.provider} defaultSort={{ key: "requests", dir: "desc" }} />
          )}
        </CardBody>
      </Card>
    </>
  );
}
