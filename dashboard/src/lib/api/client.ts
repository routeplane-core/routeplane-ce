// CE data-plane client. Talks ONLY to the Community-Edition gateway surfaces,
// authenticated with the operator's stored rp_ key. No control-plane endpoints.
import { apiUrl } from "@/lib/api/config";
import { getStoredKey } from "@/lib/auth";
import type {
  CreateProviderInput,
  CustomProvider,
  LogRow,
  ModelsResponse,
  RequestMetric,
  StatusResponse,
  UsageEvent,
} from "@/lib/api/types";

function headers(extra?: Record<string, string>): HeadersInit {
  const key = getStoredKey();
  return { ...(key ? { "x-routeplane-api-key": key } : {}), ...extra };
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), { headers: headers() });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

/**
 * Send a JSON body (POST/DELETE) and surface the gateway's OpenAI-style error
 * message (`{ error: { message } }`) on failure so the UI can show it verbatim.
 */
async function sendJSON<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method,
    headers: headers(body !== undefined ? { "Content-Type": "application/json" } : undefined),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) {
    let msg = `${path} → ${res.status}`;
    try {
      const j = JSON.parse(text);
      msg = j?.error?.message ?? msg;
    } catch {
      /* non-JSON error body — keep the status message */
    }
    throw new Error(msg);
  }
  return (text ? JSON.parse(text) : undefined) as T;
}

export const api = {
  getStatus: () => getJSON<StatusResponse>("/status"),
  getModels: () => getJSON<ModelsResponse>("/v1/models"),
  getAnalytics: () => getJSON<UsageEvent[]>("/analytics"),
  getLogs: async () => (await getJSON<{ events: LogRow[] }>("/v1/logs")).events,

  /** Parse the Prometheus `/metrics` text into per-provider request counters. */
  getMetrics: async (): Promise<RequestMetric[]> => {
    const res = await fetch(apiUrl("/metrics"), { headers: headers() });
    if (!res.ok) throw new Error(`/metrics → ${res.status}`);
    const text = await res.text();
    const out: RequestMetric[] = [];
    for (const line of text.split("\n")) {
      if (!line.startsWith("rp_requests_total")) continue;
      const m = line.match(/provider="([^"]*)".*outcome="([^"]*)"\}\s+([0-9.]+)/);
      if (m) out.push({ provider: m[1], outcome: m[2], value: Number(m[3]) });
    }
    return out;
  },

  /** Purge the exact-match response cache. */
  purgeCache: async (): Promise<void> => {
    const res = await fetch(apiUrl("/v1/cache/purge"), { method: "POST", headers: headers() });
    if (!res.ok) throw new Error(`/v1/cache/purge → ${res.status}`);
  },

  // --- Runtime custom OpenAI-compatible providers ---
  listProviders: async (): Promise<CustomProvider[]> =>
    (await getJSON<{ data: CustomProvider[] }>("/v1/providers")).data,

  /** Create/update a custom provider. The raw api_key is sent once; the response
   *  carries only the masked key. */
  createProvider: (input: CreateProviderInput): Promise<CustomProvider> =>
    sendJSON<CustomProvider>("POST", "/v1/providers", input),

  deleteProvider: (name: string): Promise<void> =>
    sendJSON<void>("DELETE", `/v1/providers/${encodeURIComponent(name)}`),
};

/**
 * Stream a chat completion for the Playground. Calls the real CE gateway and
 * yields text deltas via `onDelta`. Returns the accumulated text. Any provider
 * error is surfaced to the caller (the gateway needs a configured provider key).
 */
export async function streamChat(
  body: Record<string, unknown>,
  onDelta: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(apiUrl("/v1/chat/completions"), {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    throw new Error(detail || `gateway ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith("data:")) continue;
      const payload = t.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        const json = JSON.parse(payload);
        const delta = json?.choices?.[0]?.delta?.content;
        if (typeof delta === "string") onDelta(delta);
      } catch {
        /* keep-alive or partial line — ignore */
      }
    }
  }
}
