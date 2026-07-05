import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ScrollText, Search } from "lucide-react";
import { api } from "@/lib/api/client";
import type { LogRow } from "@/lib/api/types";
import { Card, CardBody } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/layout/PageHeader";
import { EnterpriseHint } from "@/components/EnterpriseHint";
import { DataTable, type Column } from "@/components/ui/table";
import { SegmentedControl, Pagination, KeyValueList } from "@/components/ui/misc";
import { Input } from "@/components/ui/input";
import { SkeletonRows, EmptyState, ErrorState } from "@/components/ui/states";
import { Drawer, DrawerContent, DrawerHeader, DrawerBody } from "@/components/ui/drawer";
import { formatMicroUsd, formatMs, formatRelativeTime, formatDateTime, formatCompact } from "@/lib/utils";

const PAGE = 20;

type OutcomeFilter = "all" | "success" | "error" | "blocked";

const OUTCOME_TONE = (o: LogRow["outcome"]) =>
  o === "success" ? "success" : o === "blocked" ? "warning" : "danger";

export function Logs() {
  const { data, isLoading, isError, refetch } = useQuery({ queryKey: ["logs"], queryFn: api.getLogs });
  const [outcome, setOutcome] = useState<OutcomeFilter>("all");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<LogRow | null>(null);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (data ?? []).filter((r) => {
      if (outcome !== "all" && r.outcome !== outcome) return false;
      if (!needle) return true;
      return [r.model, r.provider, r.virtual_key_name, r.id, r.error, r.use_case]
        .some((v) => v?.toLowerCase().includes(needle));
    });
  }, [data, outcome, q]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE));
  const clampedPage = Math.min(page, pageCount - 1);
  const rows = filtered.slice(clampedPage * PAGE, clampedPage * PAGE + PAGE);

  const columns: Column<LogRow>[] = [
    {
      key: "ts",
      header: "Time",
      sortValue: (l) => l.timestamp,
      cell: (l) => <span className="whitespace-nowrap text-muted-foreground">{formatRelativeTime(l.timestamp)}</span>,
    },
    { key: "key", header: "Key", cell: (l) => <span className="font-mono text-xs">{l.virtual_key_name || "—"}</span> },
    { key: "provider", header: "Provider", cell: (l) => <span className="text-xs">{l.provider}</span> },
    { key: "model", header: "Model", cell: (l) => <span className="font-mono text-xs">{l.model}</span> },
    { key: "outcome", header: "Outcome", cell: (l) => <Badge tone={OUTCOME_TONE(l.outcome)}>{l.outcome}</Badge> },
    {
      key: "tokens",
      header: "Tokens",
      align: "right",
      sortValue: (l) => l.total_tokens,
      cell: (l) => <span className="tnum text-muted-foreground">{formatCompact(l.total_tokens)}</span>,
    },
    {
      key: "latency",
      header: "Latency",
      align: "right",
      sortValue: (l) => l.latency_ms ?? 0,
      cell: (l) => <span className="tnum">{l.latency_ms != null ? formatMs(l.latency_ms) : "—"}</span>,
    },
    {
      key: "cost",
      header: "Cost",
      align: "right",
      sortValue: (l) => l.cost_micro_usd ?? 0,
      cell: (l) => <span className="tnum">{l.cost_micro_usd != null ? formatMicroUsd(l.cost_micro_usd) : "—"}</span>,
    },
  ];

  return (
    <>
      <PageHeader
        title="Logs & Traces"
        description="Recent requests from the gateway's in-memory log ring. Filter by outcome or search across key, provider, and model."
      />

      <div className="mb-4">
        <EnterpriseHint title="Per-user usage logs & attribution">
          Durable, per-user request logs — who ran each call, attributed as company (business process) vs. personal —
          with long-term retention and export. The Community Edition keeps an in-memory ring of recent requests only.
        </EnterpriseHint>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <SegmentedControl
          value={outcome}
          onChange={(v) => {
            setOutcome(v);
            setPage(0);
          }}
          options={[
            { value: "all", label: "All" },
            { value: "success", label: "Success" },
            { value: "error", label: "Error" },
            { value: "blocked", label: "Blocked" },
          ]}
        />
        <div className="relative">
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setPage(0);
            }}
            placeholder="Search key, provider, model…"
            className="w-64 pl-8"
          />
        </div>
        <span className="ml-auto text-xs text-muted-foreground">{filtered.length.toLocaleString()} requests</span>
      </div>

      <Card>
        <CardBody className="p-0">
          {isLoading ? (
            <div className="p-5">
              <SkeletonRows rows={10} />
            </div>
          ) : isError ? (
            <ErrorState message="The gateway's /v1/logs didn't respond." onRetry={() => refetch()} />
          ) : (data ?? []).length === 0 ? (
            <EmptyState
              icon={ScrollText}
              title="No requests yet"
              description="This is expected on a fresh install. Requests appear here as soon as traffic flows through the gateway."
            />
          ) : filtered.length === 0 ? (
            <EmptyState icon={ScrollText} title="No requests match" description="Try a different outcome filter or search term." />
          ) : (
            <>
              <DataTable
                columns={columns}
                rows={rows}
                getRowId={(l) => l.id}
                onRowClick={(l) => setSelected(l)}
                defaultSort={{ key: "ts", dir: "desc" }}
              />
              <Pagination page={clampedPage} pageCount={pageCount} onPage={setPage} total={filtered.length} />
            </>
          )}
        </CardBody>
      </Card>

      <Drawer open={!!selected} onOpenChange={(o) => !o && setSelected(null)}>
        <DrawerContent size="lg">
          <DrawerHeader title="Request detail" description={selected?.id ?? ""} />
          <DrawerBody>
            {selected && (
              <div className="space-y-5">
                <KeyValueList
                  items={[
                    { label: "Timestamp", value: formatDateTime(selected.timestamp) },
                    { label: "Outcome", value: <Badge tone={OUTCOME_TONE(selected.outcome)}>{selected.outcome}</Badge> },
                    { label: "Virtual key", value: <span className="font-mono text-xs">{selected.virtual_key_name || "—"}</span> },
                    { label: "Provider", value: selected.provider },
                    { label: "Model", value: <span className="font-mono text-xs">{selected.model}</span> },
                    { label: "Use case", value: selected.use_case ?? "—" },
                    { label: "Prompt tokens", value: formatCompact(selected.prompt_tokens) },
                    { label: "Completion tokens", value: formatCompact(selected.completion_tokens) },
                    { label: "Total tokens", value: formatCompact(selected.total_tokens) },
                    { label: "Latency", value: selected.latency_ms != null ? formatMs(selected.latency_ms) : "—" },
                    { label: "Cost", value: selected.cost_micro_usd != null ? formatMicroUsd(selected.cost_micro_usd) : "—" },
                    { label: "Cache", value: selected.cache_status ?? "—" },
                  ]}
                />
                {selected.error && (
                  <div>
                    <div className="mb-1.5 text-sm font-medium text-danger">Error</div>
                    <div className="rounded-md border border-danger/30 bg-danger/5 px-3 py-2 text-xs text-danger">
                      {selected.error}
                    </div>
                  </div>
                )}
              </div>
            )}
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </>
  );
}
