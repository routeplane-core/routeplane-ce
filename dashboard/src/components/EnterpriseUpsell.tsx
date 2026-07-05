import { Lock, ArrowUpRight, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardBody } from "@/components/ui/card";
import { PageHeader } from "@/components/layout/PageHeader";

const CONTACT_URL = "https://routeplane.ai/contact";

/**
 * The single surface every Enterprise-only feature renders in the Community
 * Edition. It is intentionally inert: it ships NO enterprise logic, endpoints,
 * credentials, or data — only the feature's name, a description, and a link to
 * the routeplane.ai contact page. This is the whole enterprise footprint in CE.
 */
export function EnterpriseUpsell({
  title,
  summary,
  capabilities,
}: {
  title: string;
  summary: string;
  capabilities: string[];
}) {
  return (
    <div className="mx-auto max-w-2xl">
      <PageHeader title={title} enterprise />
      <Card>
        <CardBody className="flex flex-col items-center gap-5 py-12 text-center">
          <div className="grid h-14 w-14 place-items-center rounded-full bg-primary/10 text-primary">
            <Lock size={26} />
          </div>
          <div className="space-y-2">
            <h2 className="text-lg font-semibold">Available on Routeplane Enterprise</h2>
            <p className="mx-auto max-w-md text-sm text-muted-foreground">{summary}</p>
          </div>

          <ul className="mx-auto grid max-w-md gap-2 text-left text-sm">
            {capabilities.map((c) => (
              <li key={c} className="flex items-start gap-2">
                <Check size={16} className="mt-0.5 shrink-0 text-primary" />
                <span className="text-foreground/80">{c}</span>
              </li>
            ))}
          </ul>

          <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
            <a href={CONTACT_URL} target="_blank" rel="noreferrer noopener">
              <Button>
                Contact us <ArrowUpRight size={15} />
              </Button>
            </a>
            <a href="https://routeplane.ai" target="_blank" rel="noreferrer noopener">
              <Button variant="outline">Learn more</Button>
            </a>
          </div>
          <p className="text-xs text-muted-foreground">
            The Community Edition is Apache-2.0 and fully self-hostable. Enterprise adds
            sovereignty, governance, and agentic security — hosted or self-managed.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
