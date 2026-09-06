import { CopyButton } from "@/components/ui/misc";

export function ResponseRequestId({ requestId }: { requestId: string | null }) {
  if (!requestId) return null;
  return (
    <div className="min-w-0 space-y-1 rounded-md border px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">Gateway request ID</span>
        <CopyButton value={requestId} label="Copy request ID" />
      </div>
      <code className="block break-all">{requestId}</code>
      <p className="text-muted-foreground">Search this ID in Logs &amp; Traces. Only retained rows with a recorded request ID can match.</p>
    </div>
  );
}
