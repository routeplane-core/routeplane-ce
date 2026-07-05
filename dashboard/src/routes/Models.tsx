import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Boxes, Search } from "lucide-react";
import { api } from "@/lib/api/client";
import type { ModelEntry } from "@/lib/api/types";
import { Card, CardBody } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/layout/PageHeader";
import { DataTable, type Column } from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { SkeletonRows, EmptyState, ErrorState } from "@/components/ui/states";
import { formatCompact, formatMicroUsd } from "@/lib/utils";

export function Models() {
  const { data, isLoading, isError, refetch } = useQuery({ queryKey: ["models"], queryFn: api.getModels });
  const [q, setQ] = useState("");

  const models = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const all = data?.data ?? [];
    if (!needle) return all;
    return all.filter(
      (m) => m.id.toLowerCase().includes(needle) || m.owned_by.toLowerCase().includes(needle),
    );
  }, [data, q]);

  const columns: Column<ModelEntry>[] = [
    {
      key: "id",
      header: "Model",
      sortValue: (m) => m.id,
      cell: (m) => <span className="font-mono text-xs">{m.id}</span>,
    },
    {
      key: "owned_by",
      header: "Provider",
      sortValue: (m) => m.owned_by,
      cell: (m) => <span className="text-xs">{m.owned_by}</span>,
    },
    {
      key: "context",
      header: "Context",
      align: "right",
      sortValue: (m) => m.routeplane?.context_window ?? 0,
      cell: (m) => (
        <span className="tnum text-muted-foreground">
          {m.routeplane?.context_window ? formatCompact(m.routeplane.context_window) : "—"}
        </span>
      ),
    },
    {
      key: "modalities",
      header: "Modalities",
      cell: (m) => (
        <div className="flex flex-wrap gap-1">
          {(m.routeplane?.modalities ?? []).length > 0
            ? m.routeplane!.modalities!.map((x) => (
                <Badge key={x} tone="neutral">
                  {x}
                </Badge>
              ))
            : <span className="text-xs text-muted-foreground">—</span>}
        </div>
      ),
    },
    {
      key: "capabilities",
      header: "Capabilities",
      cell: (m) => (
        <div className="flex flex-wrap gap-1">
          {(m.routeplane?.capabilities ?? []).length > 0
            ? m.routeplane!.capabilities!.map((x) => (
                <Badge key={x} tone="primary">
                  {x}
                </Badge>
              ))
            : <span className="text-xs text-muted-foreground">—</span>}
        </div>
      ),
    },
    {
      key: "input_cost",
      header: "Input /1k",
      align: "right",
      sortValue: (m) => m.routeplane?.cost?.input_per_1k_micro_usd ?? 0,
      cell: (m) => (
        <span className="tnum">
          {m.routeplane?.cost?.input_per_1k_micro_usd != null
            ? formatMicroUsd(m.routeplane.cost.input_per_1k_micro_usd)
            : "—"}
        </span>
      ),
    },
    {
      key: "output_cost",
      header: "Output /1k",
      align: "right",
      sortValue: (m) => m.routeplane?.cost?.output_per_1k_micro_usd ?? 0,
      cell: (m) => (
        <span className="tnum">
          {m.routeplane?.cost?.output_per_1k_micro_usd != null
            ? formatMicroUsd(m.routeplane.cost.output_per_1k_micro_usd)
            : "—"}
        </span>
      ),
    },
  ];

  return (
    <>
      <PageHeader
        title="Model Catalog"
        description="The read-only OpenAI-compatible catalog served by the gateway (GET /v1/models), with Routeplane cost and capability metadata."
      />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search model or provider…" className="w-72 pl-8" />
        </div>
        <span className="ml-auto text-xs text-muted-foreground">
          {(data?.data.length ?? 0).toLocaleString()} models
        </span>
      </div>

      <Card>
        <CardBody className="p-0">
          {isLoading ? (
            <div className="p-5">
              <SkeletonRows rows={8} />
            </div>
          ) : isError ? (
            <ErrorState message="The gateway's /v1/models didn't respond." onRetry={() => refetch()} />
          ) : (data?.data.length ?? 0) === 0 ? (
            <EmptyState
              icon={Boxes}
              title="No models configured"
              description="Configure providers via .env and configs/keys.json, and their models will appear in the catalog."
            />
          ) : models.length === 0 ? (
            <EmptyState icon={Boxes} title="No models match" description="Try a different search term." />
          ) : (
            <DataTable columns={columns} rows={models} getRowId={(m) => m.id} defaultSort={{ key: "id", dir: "asc" }} />
          )}
        </CardBody>
      </Card>

      <p className="mt-3 text-xs text-muted-foreground">
        Enabling / disabling models, billing overrides, and per-model routing controls are Routeplane Enterprise
        features. In the Community Edition the catalog is read-only and reflects the gateway configuration.
      </p>
    </>
  );
}
