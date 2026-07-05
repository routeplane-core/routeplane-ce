import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";

/** Consistent page title block. `enterprise` stamps the Enterprise badge. */
export function PageHeader({
  title,
  description,
  actions,
  enterprise,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  enterprise?: boolean;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
      <div>
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
          {enterprise && <Badge tone="primary">Enterprise</Badge>}
        </div>
        {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
