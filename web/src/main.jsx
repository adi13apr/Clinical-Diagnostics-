import "@/index.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MotionConfig } from "framer-motion";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "react-router/dom";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ApiError } from "@/lib/api";
import { AuthProvider } from "@/providers/AuthProvider";
import { DiagnosisProvider } from "@/providers/DiagnosisProvider";
import { ThemeProvider } from "@/providers/ThemeProvider";
import { router } from "@/router";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
      // Retry network blips, not 4xx answers (bad filter, expired upload, signed out)
      retry: (count, error) => count < 2 && !(error instanceof ApiError && error.status >= 400 && error.status < 500),
    },
  },
});

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <DiagnosisProvider>
            <MotionConfig reducedMotion="user">
              <TooltipProvider delayDuration={200}>
                <RouterProvider router={router} />
                <Toaster position="top-right" richColors closeButton />
              </TooltipProvider>
            </MotionConfig>
          </DiagnosisProvider>
        </AuthProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>,
);
