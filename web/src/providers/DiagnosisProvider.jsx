import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCreateReport, usePredict } from "@/hooks/queries";
import { DiagnosisContext, useAuth } from "@/providers/contexts";

/**
 * Holds the Diagnose page's state above the router, so a result (or a report that takes a minute)
 * survives a visit to History and back. The image stays in this browser tab as an object URL.
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

  const value = useMemo(
    () => ({
      file,
      previewUrl,
      prediction: predictMutation.data ?? null,
      predicting: predictMutation.isPending,
      predictError: predictMutation.error,
      report: reportMutation.data ?? null,
      reporting: reportMutation.isPending,
      reportError: reportMutation.error,
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
      reportMutation.data,
      reportMutation.isPending,
      reportMutation.error,
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
