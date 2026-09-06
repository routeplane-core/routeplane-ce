// CE data-plane client. Talks ONLY to the Community-Edition gateway surfaces,
// authenticated with the operator's stored rp_ key. No control-plane endpoints.
import { apiUrl } from "@/lib/api/config";
import { getStoredToken, clearSession } from "@/lib/auth";
import { validChatTarget } from "@/lib/api/chat-target";
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
  const token = getStoredToken();
  return { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...extra };
}

/** A 401 on an authed call means the session expired or was revoked — clear it
 *  so the app falls back to the login screen. */
function on401(status: number) {
  if (status === 401) clearSession();
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), { headers: headers() });
  if (!res.ok) {
    on401(res.status);
    throw new Error(`${path} → ${res.status}`);
  }
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
    on401(res.status);
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

  /** The operator's own gateway rp_ key (session-authed reveal, for copy). */
  getConsoleApiKey: (): Promise<{ name: string; key: string }> =>
    getJSON<{ name: string; key: string }>("/v1/console/api-key"),
};

/**
 * Stream a chat completion for the Playground. Calls the real CE gateway and
 * yields text deltas via `onDelta`. Requires an explicit single provider. Any provider
 * error is surfaced to the caller (the gateway needs a configured provider key).
 */
export async function streamChat(
  body: Record<string, unknown>,
  onDelta: (text: string) => void,
  signal?: AbortSignal,
  provider = "",
  onRequestId?: (requestId: string | null) => void,
): Promise<string | null> {
  if (typeof body.model !== "string" || !validChatTarget(body.model, provider)) {
    throw new Error("Enter a model and one configured provider name before sending.");
  }
  const res = await fetch(apiUrl("/v1/chat/completions"), {
    method: "POST",
    headers: headers({ "Content-Type": "application/json", "x-routeplane-provider": provider.trim() }),
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  // Both headers identify this gateway request, not a provider completion or
  // W3C trace. Publish as soon as headers arrive, including error responses.
  const requestId = res.headers.get("x-routeplane-request-id") || res.headers.get("x-routeplane-trace-id");
  onRequestId?.(requestId);
  if (!res.ok || !res.body) {
    on401(res.status);
    const detail = await res.text().catch(() => "");
    throw new Error(detail || `gateway ${res.status}`);
  }
  if (!res.headers.get("content-type")?.toLowerCase().startsWith("text/event-stream")) {
    await res.body.cancel();
    throw new Error("Gateway returned a non-streaming response. Check the gateway/upstream streaming configuration.");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
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
        if (payload === "[DONE]") return requestId;
        let json;
        try {
          json = JSON.parse(payload);
        } catch {
          throw new Error("Gateway returned an invalid streaming event.");
        }
        if (json?.error) throw new Error("Gateway reported a streaming error. Check upstream availability and retry explicitly.");
        const delta = json?.choices?.[0]?.delta?.content;
        if (typeof delta === "string") onDelta(delta);
      }
    }
    throw new Error("Stream ended before [DONE]. The partial response is retained; check upstream availability before retrying.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
