// CE data-plane response types. These mirror ONLY the Community-Edition gateway
// surfaces (/status, /analytics, /v1/logs, /v1/models, /metrics). There are no
// control-plane / enterprise shapes here by design.

/** One element of `GET /analytics` — the tenant's own recent usage events. */
export interface UsageEvent {
  timestamp: string;
  virtual_key_name: string;
  provider: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
  use_case?: string;
  success: boolean;
  error?: string;
  cache_hit?: boolean;
  cache_status?: string;
  latency_ms?: number;
  cost?: { canonical_micro_usd?: number };
}

/** One element of `GET /v1/logs` → `{ events: LogRow[] }`. */
export interface LogRow {
  id: string;
  timestamp: string;
  virtual_key_name: string;
  provider: string;
  model: string;
  outcome: "success" | "error" | "blocked";
  error?: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  latency_ms?: number;
  cost_micro_usd?: number;
  cache_status?: string;
  use_case?: string;
}

/** `GET /status` — cache stats + per-provider circuit/latency snapshot. */
export interface StatusResponse {
  cache: CacheStats;
  providers: ProviderStatus[];
}

export interface CacheStats {
  approx_bytes: number;
  entries: number;
  hit_rate: number;
  hits: number;
  misses: number;
  oversize_drops: number;
  write_drops: number;
}

export interface ProviderStatus {
  provider: string;
  circuit: "closed" | "open" | "half_open" | string;
  latency_ewma_ms: number | null;
}

/** `GET /v1/models` — the OpenAI-compatible catalog with Routeplane metadata. */
export interface ModelsResponse {
  object: string;
  data: ModelEntry[];
}

export interface ModelEntry {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  routeplane?: {
    cost?: {
      input_per_1k_micro_usd?: number;
      output_per_1k_micro_usd?: number;
      normalized_cost_param?: number;
      source?: string;
    };
    modalities?: string[];
    capabilities?: string[];
    context_window?: number;
  };
}

/** Parsed `/metrics` request counters, aggregated per provider+outcome. */
export interface RequestMetric {
  provider: string;
  outcome: string;
  value: number;
}

/**
 * A runtime custom OpenAI-compatible provider (`GET/POST /v1/providers`). The
 * `api_key` here is ALWAYS the masked form (`…last4`) — the raw key is write-only
 * and never returned by the gateway.
 */
export interface CustomProvider {
  object: string;
  name: string;
  base_url: string;
  api_key: string; // masked, e.g. "…9999"
  models: string[];
  created_at: string;
}

/** Body for `POST /v1/providers` — the raw api_key is sent once, never read back. */
export interface CreateProviderInput {
  name: string;
  base_url: string;
  api_key: string;
  models: string[];
}
