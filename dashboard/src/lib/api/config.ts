// CE gateway location. Unset → same-origin: in production the CE gateway serves
// this SPA and its own API from the same host, so relative paths just work. In
// dev, Vite proxies these paths to a local gateway (see vite.config.ts).
export const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}
