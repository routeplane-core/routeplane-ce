import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ToastProvider } from "@/components/ui/toast";
import { App } from "@/App";
import "@/index.css";

// The Console runs against a live CE gateway, so poll on a short interval and
// refetch on focus to reflect fresh traffic. Everything reads the in-memory
// observability ring, so polling is cheap.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 10_000, refetchInterval: 15_000, refetchOnWindowFocus: true, retry: 1 },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={200}>
        <ToastProvider>
          <App />
        </ToastProvider>
      </TooltipProvider>
    </QueryClientProvider>
  </StrictMode>,
);
