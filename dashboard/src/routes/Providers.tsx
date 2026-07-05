import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plug, Plus, Trash2, Loader2 } from "lucide-react";
import { api } from "@/lib/api/client";
import type { CustomProvider } from "@/lib/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Field } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogBody,
  DialogFooter,
} from "@/components/ui/dialog";
import { PageHeader } from "@/components/layout/PageHeader";
import { DataTable, type Column } from "@/components/ui/table";
import { SkeletonRows, EmptyState, ErrorState } from "@/components/ui/states";
import { useToast } from "@/components/ui/toast";
import { formatNumber, formatRelativeTime } from "@/lib/utils";

const CIRCUIT_TONE = (c: string) =>
  c === "closed" ? "success" : c === "open" ? "danger" : c === "half_open" ? "warning" : "neutral";
const CIRCUIT_LABEL = (c: string) => (c === "half_open" ? "half-open" : c);

interface ProviderRow {
  provider: string;
  circuit: string;
  models: number;
}

function AddProviderDialog({ onClose }: { onClose: () => void }) {
  const { toast } = useToast();
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [models, setModels] = useState("");
  const [error, setError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () =>
      api.createProvider({
        name: name.trim(),
        base_url: baseUrl.trim(),
        api_key: apiKey,
        models: models
          .split(",")
          .map((m) => m.trim())
          .filter(Boolean),
      }),
    onSuccess: (p) => {
      // Invalidate everything the new provider now feeds: its own list, the
      // gateway view (/status), and the model catalog (/v1/models) so the
      // Playground and Model Catalog pick up the new models immediately.
      qc.invalidateQueries({ queryKey: ["providers"] });
      qc.invalidateQueries({ queryKey: ["status"] });
      qc.invalidateQueries({ queryKey: ["models"] });
      toast({ tone: "success", title: `Provider "${p.name}" added`, description: "Its models are usable now — no restart." });
      onClose();
    },
    onError: (e: unknown) => setError(e instanceof Error ? e.message : "Failed to add provider"),
  });

  function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    create.mutate();
  }

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader
          title="Add custom provider"
          description="Any OpenAI-compatible endpoint. The API key is stored server-side and never shown again."
        />
        <form onSubmit={submit}>
          <DialogBody className="space-y-4">
            <Field label="Name" hint="Lowercase id: a-z, 0-9, - or _" htmlFor="p-name">
              <Input id="p-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="myvllm" autoFocus />
            </Field>
            <Field label="Base URL" hint="OpenAI-compatible root (no /v1). e.g. http://vllm.internal:8000" htmlFor="p-url">
              <Input id="p-url" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://api.example.com" />
            </Field>
            <Field label="API key" hint="Sent as Authorization: Bearer to the upstream. Write-only." htmlFor="p-key">
              <Input id="p-key" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-…" />
            </Field>
            <Field label="Models" hint="Comma-separated model IDs this provider serves." htmlFor="p-models" error={error ?? undefined}>
              <Input id="p-models" value={models} onChange={(e) => setModels(e.target.value)} placeholder="custom-llama, custom-embed" />
            </Field>
          </DialogBody>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={create.isPending || !name || !baseUrl || !apiKey || !models}>
              {create.isPending && <Loader2 size={15} className="animate-spin" />}
              Add provider
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function Providers() {
  const { toast } = useToast();
  const qc = useQueryClient();
  const [adding, setAdding] = useState(false);

  const custom = useQuery({ queryKey: ["providers"], queryFn: api.listProviders });
  const status = useQuery({ queryKey: ["status"], queryFn: api.getStatus });
  const models = useQuery({ queryKey: ["models"], queryFn: api.getModels });

  const del = useMutation({
    mutationFn: (name: string) => api.deleteProvider(name),
    onSuccess: (_d, name) => {
      qc.invalidateQueries({ queryKey: ["providers"] });
      qc.invalidateQueries({ queryKey: ["status"] });
      qc.invalidateQueries({ queryKey: ["models"] });
      toast({ tone: "success", title: `Provider "${name}" removed` });
    },
    onError: (e: unknown) => toast({ tone: "error", title: "Delete failed", description: e instanceof Error ? e.message : "" }),
  });

  const rows = useMemo<ProviderRow[]>(() => {
    const providers = status.data?.providers ?? [];
    const catalog = models.data?.data ?? [];
    const modelsByOwner = new Map<string, number>();
    for (const m of catalog) modelsByOwner.set(m.owned_by, (modelsByOwner.get(m.owned_by) ?? 0) + 1);
    const circuitByName = new Map(providers.map((p) => [p.provider, p.circuit]));
    const names = new Set<string>([...circuitByName.keys(), ...modelsByOwner.keys()]);
    return [...names]
      .map((name) => ({ provider: name, circuit: circuitByName.get(name) ?? "closed", models: modelsByOwner.get(name) ?? 0 }))
      .sort((a, b) => b.models - a.models || a.provider.localeCompare(b.provider));
  }, [status.data, models.data]);

  const gatewayColumns: Column<ProviderRow>[] = [
    { key: "provider", header: "Provider", sortValue: (r) => r.provider, cell: (r) => <span className="font-mono text-xs">{r.provider}</span> },
    { key: "circuit", header: "Circuit", cell: (r) => <Badge tone={CIRCUIT_TONE(r.circuit)}>{CIRCUIT_LABEL(r.circuit)}</Badge> },
    { key: "models", header: "Models", align: "right", sortValue: (r) => r.models, cell: (r) => <span className="tnum text-muted-foreground">{formatNumber(r.models)}</span> },
  ];

  const customColumns: Column<CustomProvider>[] = [
    { key: "name", header: "Name", sortValue: (r) => r.name, cell: (r) => <span className="font-mono text-xs">{r.name}</span> },
    { key: "base_url", header: "Base URL", cell: (r) => <span className="font-mono text-2xs text-muted-foreground">{r.base_url}</span> },
    {
      key: "models",
      header: "Models",
      cell: (r) => (
        <div className="flex flex-wrap gap-1">
          {r.models.map((m) => (
            <Badge key={m} tone="neutral" className="font-mono text-2xs">{m}</Badge>
          ))}
        </div>
      ),
    },
    { key: "api_key", header: "Key", cell: (r) => <span className="font-mono text-2xs text-muted-foreground">{r.api_key}</span> },
    { key: "created_at", header: "Added", cell: (r) => <span className="text-2xs text-muted-foreground">{formatRelativeTime(r.created_at)}</span> },
    {
      key: "actions",
      header: "",
      align: "right",
      cell: (r) => (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            if (window.confirm(`Remove provider "${r.name}"? Requests to its models will stop resolving.`)) del.mutate(r.name);
          }}
        >
          <Trash2 size={14} className="text-danger" />
        </Button>
      ),
    },
  ];

  return (
    <>
      <PageHeader
        title="Provider Integrations"
        description="Add custom OpenAI-compatible providers and use them immediately — no restart. Built-in providers get their keys from .env."
        actions={
          <Button onClick={() => setAdding(true)}>
            <Plus size={15} /> Add provider
          </Button>
        }
      />

      {adding && <AddProviderDialog onClose={() => setAdding(false)} />}

      <Card>
        <CardHeader title="Custom providers" description="OpenAI-compatible endpoints you've configured. Keys are stored server-side, write-only." />
        <CardBody className="p-0">
          {custom.isLoading ? (
            <div className="p-5"><SkeletonRows rows={3} /></div>
          ) : custom.isError ? (
            <ErrorState message="Couldn't load custom providers." onRetry={() => custom.refetch()} />
          ) : (custom.data?.length ?? 0) === 0 ? (
            <EmptyState
              icon={Plug}
              title="No custom providers yet"
              description="Add an OpenAI-compatible endpoint to route requests to it. Its models appear in the catalog and Playground immediately."
            />
          ) : (
            <DataTable columns={customColumns} rows={custom.data ?? []} getRowId={(r) => r.name} />
          )}
        </CardBody>
      </Card>

      <Card className="mt-4">
        <CardHeader title="All providers (as the gateway sees them)" description="Circuit state from /status and model counts from the catalog — built-in and custom." />
        <CardBody className="p-0">
          {status.isLoading || models.isLoading ? (
            <div className="p-5"><SkeletonRows rows={5} /></div>
          ) : status.isError && models.isError ? (
            <ErrorState message="The gateway didn't respond." onRetry={() => { status.refetch(); models.refetch(); }} />
          ) : rows.length === 0 ? (
            <EmptyState icon={Plug} title="No providers configured" description="Add a custom provider above, or wire built-ins via .env." />
          ) : (
            <DataTable columns={gatewayColumns} rows={rows} getRowId={(r) => r.provider} defaultSort={{ key: "models", dir: "desc" }} />
          )}
        </CardBody>
      </Card>

      <Card className="mt-4">
        <CardHeader title="How provider credentials are handled" />
        <CardBody className="space-y-2 text-sm text-muted-foreground">
          <p>
            Custom-provider API keys are submitted over this authenticated console, stored <strong>server-side and write-only</strong>
            {" "}(shown only as <span className="font-mono text-xs">…last4</span>, never returned in full), and persisted to a
            {" "}<span className="font-mono text-xs">0600</span> file the gateway reloads on boot.
          </p>
          <p>
            Built-in providers (OpenAI, Anthropic, Gemini, …) still read their keys from
            {" "}<span className="font-mono text-xs">.env</span> / <span className="font-mono text-xs">configs/keys.json</span> at boot.
          </p>
        </CardBody>
      </Card>
    </>
  );
}
