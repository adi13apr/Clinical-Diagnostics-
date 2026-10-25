/** TanStack Query hooks over lib/api.js. Keys include the user id so accounts never share cache. */
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import {
  createReport,
  getAnalytics,
  getGradcamMap,
  getHealth,
  getHistory,
  getReportJob,
  getTrace,
  predict,
} from "@/lib/api";
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

export const REPORT_POLL_MS = 2000;

/** Starts a report: resolves to a ReportJob (queued, or already done when the server has no worker). */
export function useCreateReport() {
  return useMutation({ mutationFn: (traceId) => createReport(traceId) });
}

const finished = (job) => job?.status === "done" || job?.status === "failed";

/**
 * Polls a report job every 2 s until it's done or failed. Keeps polling in a background tab, because the
 * server stores the report's trace when a poll sees the job finish.
 * @param {string|null} jobId Null disables the query.
 */
export function useReportJob(jobId) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["report-job", user?.id, jobId],
    queryFn: ({ signal }) => getReportJob(jobId, { signal }),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (finished(query.state.data) || query.state.error ? false : REPORT_POLL_MS),
    refetchIntervalInBackground: true,
    staleTime: (query) => (finished(query.state.data) ? Infinity : 0),
    // A 404 (expired or unknown job) won't change on retry; network blips and 503s might
    retry: (count, error) => error?.status !== 404 && error?.status !== 401 && count < 3,
  });
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
