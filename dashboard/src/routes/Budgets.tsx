import { Gauge, Wallet } from "lucide-react";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { PageHeader } from "@/components/layout/PageHeader";
import { CodeBlock } from "@/components/ui/misc";

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

export function Budgets() {
  return (
    <>
      <PageHeader
        title="Budgets & Limits"
        description="How the Community Edition enforces basic rate and spend limits. Limits are file-configured per key."
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Rate limits" description="Requests per minute, per key." />
          <CardBody className="space-y-2 text-sm text-muted-foreground">
            <p className="flex items-start gap-2">
              <Gauge size={15} className="mt-0.5 shrink-0 text-primary" />
              <span>
                Each key can declare a <span className="font-mono text-xs">rate_limit_rpm</span>. The gateway rejects
                requests over the limit with an HTTP 429 — enforced on the hot path with lock-free counters.
              </span>
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Spend limits" description="A cumulative spend ceiling, per key." />
          <CardBody className="space-y-2 text-sm text-muted-foreground">
            <p className="flex items-start gap-2">
              <Wallet size={15} className="mt-0.5 shrink-0 text-primary" />
              <span>
                Each key can declare a <span className="font-mono text-xs">spend_limit_micro_usd</span>. Once attributed
                cost crosses the ceiling, further requests on that key are refused.
              </span>
            </p>
          </CardBody>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader title="Where limits live" description="configs/keys.json, alongside the key definition." />
        <CardBody>
          <CodeBlock lang="json" code={KEYS_JSON_EXAMPLE} />
        </CardBody>
      </Card>

      <p className="mt-3 text-xs text-muted-foreground">
        A budget/limit management UI, hierarchical budgets across teams and business processes, alerting, and Smart
        Credits settlement are Routeplane Enterprise features. The Community Edition supports basic per-key rate and
        spend limits configured in <span className="font-mono text-xs">configs/keys.json</span>.
      </p>
    </>
  );
}
