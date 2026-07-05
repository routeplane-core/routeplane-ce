import { useState } from "react";
import { KeyRound, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { BrandLockup } from "@/components/ui/logo";
import { signIn, validateKey } from "@/lib/auth";

/**
 * CE sign-in: the operator pastes their rp_ gateway key (from configs/keys.json).
 * The key is validated against the live gateway before it is stored. No control
 * plane, no password — the key IS the credential.
 */
export function Login() {
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const ok = await validateKey(key);
      if (ok) signIn(key);
      else setError("That key was rejected by the gateway. Check configs/keys.json.");
    } catch {
      setError("Couldn't reach the CE gateway. Is it running and reachable?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-screen place-items-center bg-background p-6">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex justify-center">
          <BrandLockup sublabel="CE Console" size={40} />
        </div>
        <form onSubmit={submit} className="rounded-xl border bg-card p-6 shadow-sm">
          <h1 className="text-lg font-semibold">Connect to your gateway</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Paste a Routeplane API key from your gateway's <code className="text-xs">configs/keys.json</code>.
          </p>
          <div className="mt-4">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">Gateway key</label>
            <div className="flex items-center gap-2 rounded-md border px-2 focus-within:ring-2 focus-within:ring-ring">
              <KeyRound size={15} className="text-muted-foreground" />
              <input
                type="password"
                autoFocus
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder="rp_..."
                className="h-9 w-full bg-transparent font-mono text-sm outline-none"
              />
            </div>
          </div>
          {error && <p className="mt-3 text-sm text-danger">{error}</p>}
          <Button type="submit" className="mt-4 w-full" disabled={busy || !key.trim()}>
            {busy ? <Loader2 size={15} className="animate-spin" /> : null}
            {busy ? "Validating…" : "Connect"}
          </Button>
        </form>
        <p className="mt-4 text-center text-xs text-muted-foreground">
          Community Edition · your key is stored only in this browser
        </p>
      </div>
    </div>
  );
}
