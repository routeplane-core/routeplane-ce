import { Card, CardBody } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/layout/PageHeader";
import { CodeBlock } from "@/components/ui/misc";

interface Endpoint {
  method: "GET" | "POST";
  path: string;
  description: string;
  auth: boolean;
  body?: string;
}

const BASE = "http://localhost:8080";

const ENDPOINTS: Endpoint[] = [
  {
    method: "POST",
    path: "/v1/chat/completions",
    description: "OpenAI-compatible chat completions. Set \"stream\": true for SSE token streaming.",
    auth: true,
    body: '{"model":"MODEL_ID","messages":[{"role":"user","content":"Hello"}]}',
  },
  {
    method: "POST",
    path: "/v1/messages",
    description: "Anthropic-compatible Messages API, translated to the routed provider.",
    auth: true,
    body: '{"model":"MODEL_ID","max_tokens":256,"messages":[{"role":"user","content":"Hello"}]}',
  },
  {
    method: "POST",
    path: "/v1/embeddings",
    description: "Generate embedding vectors for one or more inputs.",
    auth: true,
    body: '{"model":"MODEL_ID","input":"The quick brown fox"}',
  },
  {
    method: "POST",
    path: "/v1/rerank",
    description: "Rerank documents against a query by relevance.",
    auth: true,
    body: '{"model":"MODEL_ID","query":"pricing","documents":["doc a","doc b"]}',
  },
  {
    method: "GET",
    path: "/v1/models",
    description: "List the available models with Routeplane cost and capability metadata.",
    auth: true,
  },
  {
    method: "GET",
    path: "/v1/logs",
    description: "Recent request logs from the gateway's in-memory observability ring.",
    auth: true,
  },
  {
    method: "GET",
    path: "/analytics",
    description: "Recent usage events (per-request tokens, cost, and outcome).",
    auth: true,
  },
  {
    method: "GET",
    path: "/status",
    description: "Exact-match cache statistics and per-provider circuit-breaker state.",
    auth: true,
  },
  {
    method: "GET",
    path: "/metrics",
    description: "Prometheus exposition of request counters and gauges.",
    auth: true,
  },
  {
    method: "GET",
    path: "/healthz",
    description: "Liveness probe. No authentication required.",
    auth: false,
  },
  {
    method: "POST",
    path: "/v1/cache/purge",
    description: "Clear the exact-match response cache.",
    auth: true,
  },
];

function curl(e: Endpoint): string {
  const lines: string[] = [];
  const verb = e.method === "GET" ? `curl ${BASE}${e.path}` : `curl -X POST ${BASE}${e.path}`;
  lines.push(verb + (e.auth || e.body ? " \\" : ""));
  if (e.auth) lines.push(`  -H "x-routeplane-api-key: $RP_API_KEY"${e.body ? " \\" : ""}`);
  if (e.body) {
    lines.push(`  -H "content-type: application/json" \\`);
    lines.push(`  -d '${e.body}'`);
  }
  return lines.join("\n");
}

const METHOD_TONE = (m: Endpoint["method"]) => (m === "GET" ? "neutral" : "primary");

export function Reference() {
  return (
    <>
      <PageHeader
        title="API Reference"
        description="The endpoints served by the Community Edition gateway. All authenticated calls send your rp_ key in the x-routeplane-api-key header."
      />

      <p className="mb-4 rounded-md border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        Parameters are forwarded to the upstream provider <span className="font-medium text-foreground">verbatim</span> (a
        faithful passthrough) — the gateway does not rewrite them. Send what the target model expects; e.g. newer
        OpenAI-family models require <span className="font-mono">max_completion_tokens</span> rather than{" "}
        <span className="font-mono">max_tokens</span>. Enterprise-only endpoints return a{" "}
        <span className="font-mono">402 enterprise_only</span> error in the Community Edition.
      </p>

      <div className="space-y-4">
        {ENDPOINTS.map((e) => (
          <Card key={`${e.method} ${e.path}`}>
            <CardBody className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={METHOD_TONE(e.method)}>{e.method}</Badge>
                <span className="font-mono text-sm">{e.path}</span>
                {!e.auth && <Badge tone="neutral">no auth</Badge>}
              </div>
              <p className="text-sm text-muted-foreground">{e.description}</p>
              <CodeBlock lang="bash" code={curl(e)} />
            </CardBody>
          </Card>
        ))}
      </div>

      <p className="mt-4 text-xs text-muted-foreground">
        Replace <span className="font-mono">MODEL_ID</span> with an id from the Model Catalog and set{" "}
        <span className="font-mono">$RP_API_KEY</span> to your gateway key. Point the base URL at your own gateway host.
      </p>
    </>
  );
}
