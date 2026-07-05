import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Community Edition Console — a static SPA served by the CE gateway itself in
// production (single Docker image). `@` resolves to `src/`.
//
// In dev, the CE data-plane endpoints are proxied to a locally running gateway
// so the SPA can call them same-origin (no CORS). Override the target with
// VITE_GATEWAY_TARGET (defaults to the CE gateway's :8080). Only the CE gateway
// surfaces are proxied — there is no control plane in CE.
const target = process.env.VITE_GATEWAY_TARGET ?? "http://localhost:8080";
const proxy = Object.fromEntries(
  ["/v1", "/status", "/analytics", "/metrics", "/healthz"].map((p) => [
    p,
    { target, changeOrigin: true },
  ]),
);

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: { port: 5273, proxy },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
});
