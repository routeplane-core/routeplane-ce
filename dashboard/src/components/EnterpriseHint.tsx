import { Lock, ArrowUpRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const CONTACT_URL = "https://routeplane.ai/contact";

/**
 * A compact, inline "this is an Enterprise capability" marker for use INSIDE a
 * CE-functional page (as opposed to <EnterpriseUpsell/>, which is a whole page).
 * Inert: renders a label, a one-line pitch, and a contact link — no enterprise
 * logic, endpoints, or data.
 */
export function EnterpriseHint({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-primary/20 bg-primary/5 p-4">
      <div className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-md bg-primary/10 text-primary">
        <Lock size={15} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{title}</span>
          <Badge tone="primary">Enterprise</Badge>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">{children}</p>
      </div>
      <a
        href={CONTACT_URL}
        target="_blank"
        rel="noreferrer noopener"
        className="mt-0.5 inline-flex shrink-0 items-center gap-1 text-xs font-medium text-primary hover:underline"
      >
        Contact us <ArrowUpRight size={13} />
      </a>
    </div>
  );
}
