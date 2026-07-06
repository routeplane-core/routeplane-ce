import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Play, Square } from "lucide-react";
import { api, streamChat } from "@/lib/api/client";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/layout/PageHeader";
import { Field, Textarea } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function Playground() {
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: api.getModels });
  const modelList = models?.data ?? [];

  const [model, setModel] = useState("");
  const [system, setSystem] = useState("You are a helpful assistant.");
  const [user, setUser] = useState("Say hello in one sentence.");
  const [output, setOutput] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  // Default the model picker to the first catalog entry once it loads.
  useEffect(() => {
    if (!model && modelList.length > 0) setModel(modelList[0].id);
  }, [model, modelList]);

  const run = async () => {
    if (!model) return;
    const ctrl = new AbortController();
    controllerRef.current = ctrl;
    setRunning(true);
    setOutput("");
    setError(null);

    const messages: { role: string; content: string }[] = [];
    if (system.trim()) messages.push({ role: "system", content: system });
    messages.push({ role: "user", content: user });

    try {
      await streamChat({ model, messages }, (t) => setOutput((o) => o + t), ctrl.signal);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        // Stopped by the user — leave whatever streamed so far.
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setRunning(false);
      controllerRef.current = null;
    }
  };

  const stop = () => controllerRef.current?.abort();

  return (
    <>
      <PageHeader
        title="Playground"
        description="Send an OpenAI-compatible chat request through the live CE gateway and watch it stream back."
      />

      <p className="mb-4 rounded-md border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        The gateway passes parameters through to the provider <span className="font-medium text-foreground">verbatim</span> —
        it doesn't rewrite them. If you build requests by hand, send what the upstream expects: newer OpenAI-family models
        require <span className="font-mono">max_completion_tokens</span> instead of <span className="font-mono">max_tokens</span>.
      </p>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Request" />
          <CardBody className="space-y-4">
            <Field label="Model">
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder={modelList.length ? "Select a model" : "No models available"} />
                </SelectTrigger>
                <SelectContent>
                  {modelList.map((m) => (
                    <SelectItem key={m.id} value={m.id}>
                      {m.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field label="System">
              <Textarea value={system} onChange={(e) => setSystem(e.target.value)} rows={2} className="font-sans" />
            </Field>
            <Field label="User">
              <Textarea value={user} onChange={(e) => setUser(e.target.value)} rows={5} className="font-sans" />
            </Field>
            {running ? (
              <Button variant="danger" onClick={stop}>
                <Square size={14} /> Stop
              </Button>
            ) : (
              <Button onClick={run} disabled={!model}>
                <Play size={15} /> Run
              </Button>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Response" description="Streamed from the selected model via the gateway." />
          <CardBody className="space-y-3">
            <div className="min-h-40 whitespace-pre-wrap rounded-md border bg-muted/30 px-3 py-2.5 text-sm leading-relaxed">
              {output || (
                <span className="text-muted-foreground">Run a request to see the streamed response.</span>
              )}
              {running && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-primary align-middle" />
              )}
            </div>
            {error && (
              <div className="flex items-start gap-2 rounded-md border border-danger/30 bg-danger/5 px-3 py-2.5 text-xs text-danger">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <div className="font-medium">Gateway error</div>
                  <div className="mt-0.5 break-words font-mono">{error}</div>
                  <div className="mt-1 text-danger/80">
                    A common cause is a provider key that isn't configured for the selected model.
                  </div>
                </div>
              </div>
            )}
          </CardBody>
        </Card>
      </div>
    </>
  );
}
