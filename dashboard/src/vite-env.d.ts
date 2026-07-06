/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** CE gateway base URL. Unset → same-origin (the CE gateway serves this SPA). */
  readonly VITE_API_BASE?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
