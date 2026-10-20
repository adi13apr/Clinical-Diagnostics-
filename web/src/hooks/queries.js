/** TanStack Query hooks over lib/api.js. Keys include the user id so accounts never share cache. */
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { createReport, getAnalytics, getGradcamMap, getHealth, getHistory, getTrace, predict } from "@/lib/api";
import { useAuth } from "@/providers/contexts";

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: ({ signal }) => getHealth({ signal }), staleTime: 60_000 });
}

export function usePredict() {
  return useMutation({ mutationFn: (file) => predict(file) });
}

/** Grad-CAM map for one prediction; cached for the session because the upload is fixed. */
export function useGradcamMap(traceId) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["gradcam-map", user?.id, traceId],
    queryFn: ({ signal }) => getGradcamMap(traceId, { signal }),
    enabled: Boolean(traceId),
    staleTime: Infinity,
  });
}

export function useCreateReport() {
  return useMutation({ mutationFn: (traceId) => createReport(traceId) });
}

export function useHistory(filters) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["history", user?.id, filters],
    queryFn: ({ signal }) => getHistory(filters, { signal }),
    placeholderData: keepPreviousData, // keep the table while the next page loads
  });
}

export function useAnalytics(days) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["analytics", user?.id, days],
    queryFn: ({ signal }) => getAnalytics({ days }, { signal }),
  });
}

export function useTrace(traceId) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["trace", user?.id, traceId],
    queryFn: ({ signal }) => getTrace(traceId, { signal }),
    enabled: Boolean(traceId),
  });
}
