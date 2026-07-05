// CE Console auth.
//
// The Community Edition gateway is single-tenant and key-authenticated: there is
// no control plane, no login server, no SSO. The Console therefore gates behind a
// single screen where the operator pastes their `rp_` gateway key (from
// configs/keys.json). The key is validated against the live gateway, stored in
// localStorage, and sent as `x-routeplane-api-key` on every request. The only
// thing in storage is the operator's OWN key — never any other secret.

import { useSyncExternalStore } from "react";
import { apiUrl } from "@/lib/api/config";

const KEY_STORAGE = "rp_ce_key";

const listeners = new Set<() => void>();
function emit() {
  for (const l of listeners) l();
}
function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

/** The stored rp_ key sent on every gateway request (or null when signed out). */
export function getStoredKey(): string | null {
  try {
    return localStorage.getItem(KEY_STORAGE);
  } catch {
    return null;
  }
}

function readAuthed(): boolean {
  return Boolean(getStoredKey());
}

/** Persist the validated key. */
export function signIn(key: string) {
  try {
    localStorage.setItem(KEY_STORAGE, key.trim());
  } catch {
    /* storage unavailable — in-memory listeners still fire */
  }
  emit();
}

/** Clear the credential (Sign out). */
export function signOut() {
  try {
    localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* ignore */
  }
  emit();
}

/** Reactive logged-in flag — true iff an rp_ key is stored. */
export function useAuthed(): boolean {
  return useSyncExternalStore(subscribe, readAuthed, () => false);
}

/**
 * Validate a key against the live CE gateway. `GET /v1/models` returns 200 for a
 * valid key and 401 for an invalid one. Returns true on success; throws on a
 * network/other error so the caller can distinguish "can't reach gateway" from
 * "invalid key".
 */
export async function validateKey(key: string): Promise<boolean> {
  const res = await fetch(apiUrl("/v1/models"), {
    headers: { "x-routeplane-api-key": key.trim() },
  });
  if (res.status === 401 || res.status === 403) return false;
  if (!res.ok) throw new Error(`gateway ${res.status}`);
  return true;
}
