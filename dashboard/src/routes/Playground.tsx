import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Play, Square } from "lucide-react";
import { api, streamChat } from "@/lib/api/client";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/layout/PageHeader";
import { Field, Input, Textarea } from "@/components/ui/input";
import { validChatTarget } from "@/lib/api/chat-target";
import { ResponseRequestId } from "@/components/ui/request-id";

export function Playground() {
  const { data: models, isError: catalogError } = useQuery({ queryKey: ["models"], queryFn: api.getModels });
  const { data: status } = useQuery({ queryKey: ["status"], queryFn: api.getStatus });
  const modelList = models?.data ?? [];

  const [model, setModel] = useState("");
  const [provider, setProvider] = useState("");
  const [system, setSystem] = useState("You are a helpful assistant.");
  const [user, setUser] = useState("Say hello in one sentence.");
  const [output, setOutput] = useState("");
  const [requestId, setRequestId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const run = async () => {
    if (!validChatTarget(model, provider)) return;
    const ctrl = new AbortController();
    controllerRef.current = ctrl;
    setRunning(true);
    setOutput("");
    setRequestId(null);
    setError(null);

    const messages: { role: string; content: string }[] = [];
    if (system.trim()) messages.push({ role: "system", content: system });
    messages.push({ role: "user", content: user });

    try {
      await streamChat({ model: model.trim(), messages }, (t) => setOutput((o) => o + t), ctrl.signal, provider.trim(), setRequestId);
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
        Catalogue entries are suggestions, not proof that a model is runnable with your credentials.
        Enter the exact model and configured provider name. For a self-hosted model, use
        <span className="font-mono"> self_hosted</span>; the operator configures its URL on the gateway, not here.
      </p>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Request" />
          <CardBody className="space-y-4">
            <Field label="Model" htmlFor="chat-model" hint={catalogError ? "Catalogue unavailable. You can still enter a model configured by your operator." : modelList.length === 0 ? "No catalogue entries. Enter your self-hosted model explicitly." : "Choose a suggestion or enter a model explicitly."}>
              <Input id="chat-model" list="chat-models" value={model} onChange={(e) => setModel(e.target.value)} placeholder="Exact model name" disabled={running} />
              <datalist id="chat-models">{modelList.map((m) => <option key={m.id} value={m.id} />)}</datalist>
            </Field>
            <Field label="Provider" htmlFor="chat-provider" hint="One configured provider name, for example self_hosted. No URLs or fallback chains.">
              <Input id="chat-provider" list="chat-providers" value={provider} onChange={(e) => setProvider(e.target.value)} placeholder="Configured provider name" disabled={running} />
              <datalist id="chat-providers">{status?.providers?.map((p) => <option key={p.provider} value={p.provider} />)}</datalist>
            </Field>
            <Field label="System" htmlFor="chat-system">
              <Textarea id="chat-system" value={system} onChange={(e) => setSystem(e.target.value)} rows={2} className="font-sans" />
            </Field>
            <Field label="User" htmlFor="chat-user">
              <Textarea id="chat-user" value={user} onChange={(e) => setUser(e.target.value)} rows={5} className="font-sans" />
            </Field>
            {running ? (
              <Button variant="danger" onClick={stop}>
                <Square size={14} /> Stop
              </Button>
            ) : (
              <Button onClick={run} disabled={!validChatTarget(model, provider)}>
                <Play size={15} /> Run
              </Button>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Response" description="Streamed from the selected model via the gateway." />
          <CardBody className="space-y-3">
            <ResponseRequestId requestId={requestId} />
            <div aria-label="Streamed response" className="min-h-40 whitespace-pre-wrap break-words [overflow-wrap:anywhere] rounded-md border bg-muted/30 px-3 py-2.5 text-sm leading-relaxed">
              {output || (
                <span className="text-muted-foreground">Run a request to see the streamed response.</span>
              )}
              {running && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-primary align-middle" />
              )}
            </div>
            {error && (
              <div role="alert" className="flex items-start gap-2 rounded-md border border-danger/30 bg-danger/5 px-3 py-2.5 text-xs text-danger">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <div className="font-medium">Gateway error</div>
                  <div className="mt-0.5 break-words font-mono">{error}</div>
                  <div className="mt-1 text-danger/80">
                    Check the model and provider names with your operator, then check that provider's credentials and upstream availability. No fallback was selected by this UI.
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
