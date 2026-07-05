import { useState } from "react";
import { Mail, Lock, Loader2, KeyRound, ShieldCheck, ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BrandLockup } from "@/components/ui/logo";
import { login, signup } from "@/lib/auth";

const CONTACT_URL = "https://routeplane.ai/contact";
type Mode = "login" | "signup";

export function Login() {
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isSignup = mode === "signup";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (isSignup) {
      if (password.length < 10) return setError("Password must be at least 10 characters.");
      if (password !== confirm) return setError("Passwords don't match.");
    }
    setBusy(true);
    try {
      if (isSignup) await signup(email, password);
      else await login(email, password);
      // On success the stored session flips useAuthed() → the app renders.
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  function swap(next: Mode) {
    setMode(next);
    setError(null);
    setPassword("");
    setConfirm("");
  }

  return (
    <div className="grid min-h-screen place-items-center bg-background p-6">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex justify-center">
          <BrandLockup sublabel="CE Console" size={40} />
        </div>

        <div className="rounded-xl border bg-card p-6 shadow-sm">
          <h1 className="text-lg font-semibold">{isSignup ? "Create your account" : "Sign in"}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {isSignup ? "Set up an account for this gateway's console." : "Use your email and password."}
          </p>

          <form onSubmit={submit} className="mt-4 space-y-3">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-muted-foreground">Email</span>
              <div className="flex items-center gap-2 rounded-md border px-2 focus-within:ring-2 focus-within:ring-ring">
                <Mail size={15} className="text-muted-foreground" />
                <input
                  type="email"
                  autoComplete="email"
                  autoFocus
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className="h-9 w-full bg-transparent text-sm outline-none"
                />
              </div>
            </label>

            <label className="block">
              <span className="mb-1 block text-xs font-medium text-muted-foreground">Password</span>
              <div className="flex items-center gap-2 rounded-md border px-2 focus-within:ring-2 focus-within:ring-ring">
                <Lock size={15} className="text-muted-foreground" />
                <input
                  type="password"
                  autoComplete={isSignup ? "new-password" : "current-password"}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={isSignup ? "At least 10 characters" : "••••••••"}
                  className="h-9 w-full bg-transparent text-sm outline-none"
                />
              </div>
            </label>

            {isSignup && (
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-muted-foreground">Confirm password</span>
                <div className="flex items-center gap-2 rounded-md border px-2 focus-within:ring-2 focus-within:ring-ring">
                  <Lock size={15} className="text-muted-foreground" />
                  <input
                    type="password"
                    autoComplete="new-password"
                    required
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                    placeholder="••••••••"
                    className="h-9 w-full bg-transparent text-sm outline-none"
                  />
                </div>
              </label>
            )}

            {error && <p className="text-sm text-danger">{error}</p>}

            <Button type="submit" className="w-full" disabled={busy || !email || !password}>
              {busy && <Loader2 size={15} className="animate-spin" />}
              {busy ? (isSignup ? "Creating…" : "Signing in…") : isSignup ? "Create account" : "Sign in"}
            </Button>
          </form>

          {/* Enterprise-only sign-in options (inert in CE → contact link). */}
          <div className="mt-4 border-t pt-4">
            <a href={CONTACT_URL} target="_blank" rel="noreferrer noopener" className="block">
              <button
                type="button"
                className="flex w-full items-center justify-center gap-2 rounded-md border border-dashed px-3 py-2 text-sm text-muted-foreground hover:bg-muted"
              >
                <KeyRound size={15} /> Sign in with SSO <Badge tone="primary">Enterprise</Badge>
              </button>
            </a>
            <p className="mt-2 flex items-center justify-center gap-1.5 text-xs text-muted-foreground">
              <ShieldCheck size={13} /> Two-factor authentication & SSO/SCIM are available on
              <a href={CONTACT_URL} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-0.5 font-medium text-primary hover:underline">
                Enterprise <ArrowUpRight size={11} />
              </a>
            </p>
          </div>

          <p className="mt-4 text-center text-sm text-muted-foreground">
            {isSignup ? "Already have an account?" : "No account yet?"}{" "}
            <button type="button" onClick={() => swap(isSignup ? "login" : "signup")} className="font-medium text-primary hover:underline">
              {isSignup ? "Sign in" : "Create one"}
            </button>
          </p>
        </div>

        <p className="mt-4 text-center text-xs text-muted-foreground">
          Community Edition · your session is stored only in this browser
        </p>
      </div>
    </div>
  );
}
