import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Eye, EyeOff, KeyRound } from "lucide-react";
import { api } from "@/lib/api/client";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/layout/PageHeader";
import { EnterpriseHint } from "@/components/EnterpriseHint";
import { CopyButton, CodeBlock } from "@/components/ui/misc";
import { SkeletonRows, ErrorState } from "@/components/ui/states";

const KEYS_JSON_EXAMPLE = `{
  "keys": [
    {
      "name": "prod-app",
      "key": "rp_...",
      "rate_limit_rpm": 600,
      "spend_limit_micro_usd": 50000000
    }
  ]
}`;

export function Keys() {
  const [reveal, setReveal] = useState(false);
  const apiKey = useQuery({ queryKey: ["console-api-key"], queryFn: api.getConsoleApiKey });

  const key = apiKey.data?.key ?? "";
  const masked = key ? `${key.slice(0, 6)}${"•".repeat(Math.max(key.length - 10, 4))}${key.slice(-4)}` : "";

  return (
    <>
      <PageHeader
        title="API Keys"
        description="Your Routeplane gateway key — use it as x-routeplane-api-key when calling the gateway from your app or SDK."
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Your gateway key" description="The rp_ key this gateway authenticates callers with." />
          <CardBody className="space-y-3">
            {apiKey.isLoading ? (
              <SkeletonRows rows={2} />
            ) : apiKey.isError ? (
              <ErrorState message="Couldn't load your gateway key." onRetry={() => apiKey.refetch()} />
            ) : (
              <>
                {apiKey.data?.name && (
                  <div className="text-xs text-muted-foreground">
                    Name: <span className="font-medium text-foreground">{apiKey.data.name}</span>
                  </div>
                )}
                <div className="flex items-center gap-2 rounded-md border bg-muted/30 px-3 py-2">
                  <KeyRound size={14} className="shrink-0 text-muted-foreground" />
                  <span className="min-w-0 flex-1 truncate font-mono text-sm">{reveal ? key : masked}</span>
                  <Button variant="ghost" size="sm" onClick={() => setReveal((r) => !r)} aria-label={reveal ? "Hide key" : "Reveal key"}>
                    {reveal ? <EyeOff size={14} /> : <Eye size={14} />}
                  </Button>
                  <CopyButton value={key} label="Copy" />
                </div>
                <p className="text-xs text-muted-foreground">
                  Send it as <span className="font-mono">x-routeplane-api-key</span> (or{" "}
                  <span className="font-mono">Authorization: Bearer</span>) on requests to this gateway. Treat it like a
                  password — anyone with it can spend against your providers.
                </p>
              </>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Keys for API callers" description="configs/keys.json, loaded by the gateway at boot." />
          <CardBody className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Community Edition keys are declared in <span className="font-mono text-xs">configs/keys.json</span> next to
              the gateway, each with a name and optional basic rate / spend limits.
            </p>
            <CodeBlock lang="json" code={KEYS_JSON_EXAMPLE} />
          </CardBody>
        </Card>
      </div>

      <div className="mt-4">
        <EnterpriseHint title="Multiple API keys with per-key usage tracking">
          Issue and rotate multiple scoped virtual keys — one per app, team, or environment — each with its own usage,
          spend, and rate limits tracked independently. The Community Edition uses a single file-configured key.
        </EnterpriseHint>
      </div>
    </>
  );
}
