import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Database, DatabaseZap, HardDrive, Trash2, TrendingUp, XCircle } from "lucide-react";
import { api } from "@/lib/api/client";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/ui/stat-card";
import { SkeletonRows, ErrorState } from "@/components/ui/states";
import { useToast } from "@/components/ui/toast";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogBody,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { formatCompact, formatNumber, formatPercent } from "@/lib/utils";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(1)} ${units[i]}`;
}

export function Cache() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { data, isLoading, isError, refetch } = useQuery({ queryKey: ["status"], queryFn: api.getStatus });
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [purging, setPurging] = useState(false);

  const cache = data?.cache;

  const purge = async () => {
    setPurging(true);
    try {
      await api.purgeCache();
      toast({ tone: "success", title: "Cache purged", description: "The exact-match response cache was cleared." });
      setConfirmOpen(false);
      await queryClient.invalidateQueries({ queryKey: ["status"] });
    } catch (e) {
      toast({
        tone: "error",
        title: "Purge failed",
        description: e instanceof Error ? e.message : "The gateway rejected the purge request.",
      });
    } finally {
      setPurging(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Cache"
        description="The gateway's exact-match response cache. Identical requests are served from memory, cutting cost and latency."
        actions={
          <Button variant="danger" onClick={() => setConfirmOpen(true)} disabled={!cache}>
            <Trash2 size={15} /> Purge cache
          </Button>
        }
      />

      {isError ? (
        <Card>
          <CardBody>
            <ErrorState message="The gateway's /status didn't respond." onRetry={() => refetch()} />
          </CardBody>
        </Card>
      ) : isLoading ? (
        <Card>
          <CardBody>
            <SkeletonRows rows={4} />
          </CardBody>
        </Card>
      ) : cache ? (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard icon={TrendingUp} label="Hit rate" value={formatPercent(cache.hit_rate)} sub={`${formatNumber(cache.hits)} hits · ${formatNumber(cache.misses)} misses`} />
            <StatCard icon={DatabaseZap} label="Entries" value={formatNumber(cache.entries)} sub="cached responses" />
            <StatCard icon={HardDrive} label="Size" value={formatBytes(cache.approx_bytes)} sub="approximate" />
            <StatCard icon={XCircle} label="Drops" value={formatCompact(cache.oversize_drops + cache.write_drops)} sub={`${formatNumber(cache.oversize_drops)} oversize · ${formatNumber(cache.write_drops)} write`} />
          </div>

          <Card className="mt-4">
            <CardHeader title="How the exact-match cache works" />
            <CardBody className="space-y-3 text-sm text-muted-foreground">
              <p>
                Community Edition ships a deterministic <span className="font-medium text-foreground">exact-match</span> cache:
                requests with identical parameters return a stored response without hitting the provider. Hits, misses,
                and evictions are counted in the numbers above.
              </p>
              <p className="flex items-start gap-2 rounded-md border bg-muted/30 px-3 py-2.5 text-xs">
                <Database size={14} className="mt-0.5 shrink-0" />
                <span>
                  Semantic caching (embedding-based similarity matching that also serves near-duplicate prompts) is a
                  Routeplane Enterprise feature and is not part of the Community Edition.
                </span>
              </p>
            </CardBody>
          </Card>
        </>
      ) : null}

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent size="sm">
          <DialogHeader title="Purge the cache?" description="This clears every cached response. It cannot be undone." />
          <DialogBody>
            <p className="text-sm text-muted-foreground">
              After purging, the next request for each key will miss the cache and be served by the provider until the
              cache warms again.
            </p>
          </DialogBody>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" disabled={purging}>
                Cancel
              </Button>
            </DialogClose>
            <Button variant="danger" onClick={purge} disabled={purging}>
              {purging ? "Purging…" : "Purge cache"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
