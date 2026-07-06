// CE Console auth — email + password accounts with server-issued sessions.
//
// The Community Edition gateway holds the console accounts (argon2id) and issues
// a signed session (JWT) on signup/login. The browser stores ONLY that session
// token and sends it as `Authorization: Bearer <token>`; it never sees or holds
// the gateway's rp_ key (the gateway maps a valid session to its configured key
// server-side). 2FA and SSO are Enterprise and are not part of CE.

import { useSyncExternalStore } from "react";
import { apiUrl } from "@/lib/api/config";

const TOKEN_STORAGE = "rp_ce_session";

interface SessionResponse {
  email: string;
  token: string;
  token_type: string;
  expires_in: number;
}

const listeners = new Set<() => void>();
function emit() {
  for (const l of listeners) l();
}
function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

/** The session JWT sent as `Authorization: Bearer` on every gateway call. */
export function getStoredToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_STORAGE);
  } catch {
    return null;
  }
}

function store(token: string) {
  try {
    localStorage.setItem(TOKEN_STORAGE, token);
  } catch {
    /* storage unavailable — in-memory listeners still fire */
  }
  emit();
}

/** Clear the session locally (no network). Used on 401 and by signOut. */
export function clearSession() {
  try {
    localStorage.removeItem(TOKEN_STORAGE);
  } catch {
    /* ignore */
  }
  emit();
}

function readAuthed(): boolean {
  return Boolean(getStoredToken());
}

/** Reactive logged-in flag — true iff a session token is stored. */
export function useAuthed(): boolean {
  return useSyncExternalStore(subscribe, readAuthed, () => false);
}

async function post(path: string, body: unknown): Promise<SessionResponse> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      msg = JSON.parse(text)?.error?.message ?? msg;
    } catch {
      /* keep status message */
    }
    throw new Error(msg);
  }
  return JSON.parse(text) as SessionResponse;
}

/** Create an account (email + password ≥ 10 chars). Auto-logs in on success. */
export async function signup(email: string, password: string): Promise<void> {
  const s = await post("/v1/console/signup", { email: email.trim().toLowerCase(), password });
  store(s.token);
}

/** Log in with email + password. */
export async function login(email: string, password: string): Promise<void> {
  const s = await post("/v1/console/login", { email: email.trim().toLowerCase(), password });
  store(s.token);
}

/** Sign out: best-effort server revocation, then clear locally. */
export async function signOut(): Promise<void> {
  const token = getStoredToken();
  if (token) {
    try {
      await fetch(apiUrl("/v1/console/logout"), {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch {
      /* revoke best-effort — clear locally regardless */
    }
  }
  clearSession();
}

/** The signed-in account (or null). */
export async function fetchMe(): Promise<{ email: string; created_at: string } | null> {
  const token = getStoredToken();
  if (!token) return null;
  const res = await fetch(apiUrl("/v1/console/me"), { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) return null;
  return res.json();
}
