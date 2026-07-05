import { KeyRound } from "lucide-react";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { PageHeader } from "@/components/layout/PageHeader";
import { KeyValueList, CodeBlock } from "@/components/ui/misc";
import { getStoredKey } from "@/lib/auth";

function maskKey(k: string | null): string {
  if (!k) return "—";
  return k.length <= 10 ? k : `${k.slice(0, 7)}…${k.slice(-4)}`;
}

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
  const key = getStoredKey();

  return (
    <>
      <PageHeader
        title="API Keys"
        description="How the Community Edition authenticates callers. Keys are file-configured — there is no runtime key-management API in CE."
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Connected key" description="The rp_ key this Console is using, sent as x-routeplane-api-key." />
          <CardBody>
            <KeyValueList
              items={[
                {
                  label: "Key",
                  value: (
                    <span className="inline-flex items-center gap-2 font-mono text-xs">
                      <KeyRound size={13} className="text-muted-foreground" />
                      {maskKey(key)}
                    </span>
                  ),
                },
                { label: "Header", value: <span className="font-mono text-xs">x-routeplane-api-key</span> },
                { label: "Prefix", value: <span className="font-mono text-xs">rp_</span> },
              ]}
            />
            <p className="mt-3 text-xs text-muted-foreground">
              Only your own key is ever stored (in this browser) — never any other secret. Sign out from Settings to
              clear it.
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Where keys live" description="configs/keys.json, loaded by the gateway at boot." />
          <CardBody className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Community Edition keys are declared in <span className="font-mono text-xs">configs/keys.json</span> next to
              the gateway. Each key carries a name and optional basic rate / spend limits. Editing the file and
              restarting the gateway applies the change.
            </p>
            <CodeBlock lang="json" code={KEYS_JSON_EXAMPLE} />
          </CardBody>
        </Card>
      </div>

      <p className="mt-3 text-xs text-muted-foreground">
        Runtime key issuance, rotation, and revocation — plus scoped virtual keys and per-key governance — are
        Routeplane Enterprise features managed through the control plane. The Community Edition uses file-configured
        keys only.
      </p>
    </>
  );
}
