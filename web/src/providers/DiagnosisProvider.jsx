import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCreateReport, usePredict, useReportJob } from "@/hooks/queries";
import { DiagnosisContext, useAuth } from "@/providers/contexts";

/**
 * Holds the Diagnose page's state above the router, so a result (or a report that takes a minute)
 * survives a visit to History and back. The image stays in this browser tab as an object URL.
 *
 * A report is a job: the POST returns it queued (polled here until done or failed), or already done when
 * the server writes reports itself.
 */
export function DiagnosisProvider({ children }) {
  const { user } = useAuth();
  const predictMutation = usePredict();
  const reportMutation = useCreateReport();
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [opacity, setOpacity] = useState(45);
  const [patientName, setPatientName] = useState("");
  const userId = useRef(user?.id);

  const { reset: resetPredict, mutate: runPredict } = predictMutation;
  const { reset: resetReport, mutate: runReport } = reportMutation;

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const reset = useCallback(() => {
    setFile(null);
    setPreviewUrl(null);
    setPatientName("");
    resetPredict();
    resetReport();
  }, [resetPredict, resetReport]);

  // A different account never sees the previous account's upload
  useEffect(() => {
    if (userId.current !== user?.id) {
      userId.current = user?.id;
      reset();
    }
  }, [user?.id, reset]);

  const analyze = useCallback(
    (next) => {
      resetReport();
      setFile(next);
      setPreviewUrl(URL.createObjectURL(next));
      runPredict(next);
    },
    [runPredict, resetReport],
  );

  const retry = useCallback(() => file && runPredict(file), [file, runPredict]);

  const traceId = predictMutation.data?.trace_id;
  const generateReport = useCallback(() => traceId && runReport(traceId), [traceId, runReport]);

  const started = reportMutation.data; // ReportJob from the POST
  const pollId = started?.job_id && !started.result && started.status !== "failed" ? started.job_id : null;
  const jobQuery = useReportJob(pollId);
  const job = pollId ? (jobQuery.data ?? started) : started;
  const reportStatus = reportMutation.isPending ? "submitting" : (job?.status ?? null);
  const reportFailure = job?.status === "failed" ? job.error : null;
  const reportError = useMemo(() => {
    if (reportMutation.error) return reportMutation.error;
    if (jobQuery.error) {
      return jobQuery.error.status === 404
        ? new Error("The report request expired on the server. Please try again.")
        : jobQuery.error;
    }
    if (job?.status === "failed") return new Error(reportFailure || "The report could not be written.");
    return null;
  }, [reportMutation.error, jobQuery.error, job?.status, reportFailure]);
  const report = job?.status === "done" ? job.result : null;
  const waiting = reportStatus === "submitting" || reportStatus === "queued" || reportStatus === "running";
  // "done" without a result: finished on the server, and the poll that fetches it is on its way
  const reporting = !reportError && (waiting || (reportStatus === "done" && !report));

  const value = useMemo(
    () => ({
      file,
      previewUrl,
      prediction: predictMutation.data ?? null,
      predicting: predictMutation.isPending,
      predictError: predictMutation.error,
      report,
      reporting,
      reportStatus,
      reportStartedAt: reportMutation.submittedAt || null,
      reportError,
      opacity,
      setOpacity,
      patientName,
      setPatientName,
      analyze,
      retry,
      reset,
      generateReport,
    }),
    [
      file,
      previewUrl,
      predictMutation.data,
      predictMutation.isPending,
      predictMutation.error,
      report,
      reporting,
      reportStatus,
      reportMutation.submittedAt,
      reportError,
      opacity,
      patientName,
      analyze,
      retry,
      reset,
      generateReport,
    ],
  );
  return <DiagnosisContext.Provider value={value}>{children}</DiagnosisContext.Provider>;
}
