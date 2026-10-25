import { AnimatePresence, motion } from "framer-motion";
import { Check, Download, FileText, Info, Loader2, RotateCw, Sparkles, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { useGradcamMap, useHealth } from "@/hooks/queries";
import { compositeDataUrl } from "@/lib/colormap";
import { useDiagnosis } from "@/providers/contexts";

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not read the image for the PDF"));
    img.src = src;
  });
}

function ReportBody({ result }) {
  const r = result.report;
  return (
    <motion.article
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="space-y-5"
      aria-label="AI report"
    >
      <section>
        <h3 className="mb-1.5 text-sm font-semibold">Summary</h3>
        <p className="leading-relaxed">{r.summary}</p>
      </section>
      <section className="rounded-xl bg-accent/60 p-4">
        <h3 className="mb-1.5 text-sm font-semibold text-accent-foreground">Recommendation</h3>
        <p className="leading-relaxed">{r.recommendation}</p>
      </section>
      <section>
        <h3 className="mb-1.5 text-sm font-semibold">Limitations</h3>
        <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground marker:text-primary">
          {r.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
      <footer className="space-y-2 border-t pt-4 text-xs text-muted-foreground">
        <p className="flex gap-1.5">
          <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
          <span>
            {result.source_line}
            {result.cached && " (reused from an earlier identical image)"}
          </span>
        </p>
        {result.source === "fallback" && result.reason && <p>Why the template was used: {result.reason}</p>}
        <p>{r.disclaimer}</p>
      </footer>
    </motion.article>
  );
}

/** Whole seconds since `since` (ms timestamp), updated every second. Starts at 0 when mounted. */
function useElapsedSeconds(since) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!since) return undefined;
    const id = setInterval(() => setSeconds(Math.max(0, Math.floor((Date.now() - since) / 1000))), 1000);
    return () => clearInterval(id);
  }, [since]);
  return seconds;
}

function formatElapsed(seconds) {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

const STEPS = [
  { key: "queued", label: "Queued" },
  { key: "running", label: "Writing" },
  { key: "done", label: "Checked" },
];

/** Queued -> Writing -> Checked, for reports written by a worker. */
function JobSteps({ status }) {
  const current = STEPS.findIndex((step) => step.key === (status === "done" ? "done" : status));
  return (
    <ol className="flex items-center gap-2 text-xs" aria-label="Report progress">
      {STEPS.map((step, i) => {
        const state = i < current ? "complete" : i === current ? "current" : "upcoming";
        return (
          <li key={step.key} className="flex items-center gap-2" aria-current={state === "current" ? "step" : undefined}>
            {i > 0 && <span className="h-px w-5 bg-border" aria-hidden="true" />}
            <span
              className={
                state === "upcoming"
                  ? "flex items-center gap-1 text-muted-foreground"
                  : "flex items-center gap-1 font-medium text-foreground"
              }
            >
              {state === "complete" ? (
                <Check className="size-3.5 text-primary" aria-hidden="true" />
              ) : state === "current" ? (
                <Loader2 className="size-3.5 animate-spin text-primary" aria-hidden="true" />
              ) : (
                <span className="size-1.5 rounded-full bg-muted-foreground/50" aria-hidden="true" />
              )}
              {step.label}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function progressText(status, inline, workers) {
  if (status === "queued") {
    return workers === 0
      ? "Waiting for a report worker. None is running on the server right now; the request waits up to 10 minutes."
      : "Waiting for a report worker.";
  }
  if (status === "submitting" && !inline) return "Sending the request.";
  return "Writing the report.";
}

function ReportProgress({ status, startedAt, inline, workers }) {
  const elapsed = useElapsedSeconds(startedAt);
  const queued = status === "queued" || status === "running" || status === "done";
  return (
    <div className="space-y-4">
      {queued && <JobSteps status={status} />}
      <div className="flex items-start justify-between gap-3 text-sm text-muted-foreground">
        <p className="flex items-start gap-2" role="status">
          <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin" aria-hidden="true" />
          <span>
            {progressText(status, inline, workers)} On the free LLM tier this can take up to a minute; you can visit
            other pages meanwhile.
          </span>
        </p>
        {/* Outside the live region, so screen readers aren't told every second */}
        <span className="shrink-0 tabular-nums" aria-label={`${elapsed} seconds elapsed`}>
          {formatElapsed(elapsed)}
        </span>
      </div>
      <Skeleton className="h-4 w-11/12" />
      <Skeleton className="h-4 w-10/12" />
      <Skeleton className="h-16 w-full rounded-xl" />
      <Skeleton className="h-4 w-8/12" />
      <Skeleton className="h-4 w-9/12" />
    </div>
  );
}

/** AI report (generated on request) and the PDF download, which works with or without it. */
export function ReportPanel() {
  const {
    prediction,
    report,
    reporting,
    reportStatus,
    reportStartedAt,
    reportError,
    generateReport,
    previewUrl,
    opacity,
    patientName,
    setPatientName,
  } = useDiagnosis();
  const health = useHealth();
  const map = useGradcamMap(prediction?.trace_id);
  const [exporting, setExporting] = useState(false);

  async function downloadPdf() {
    setExporting(true);
    try {
      const [{ downloadReportPdf }, img] = await Promise.all([import("@/lib/pdf"), loadImage(previewUrl)]);
      const image = compositeDataUrl(img, map.data?.cam ?? null, map.data ? Math.max(opacity, 35) / 100 : 0);
      await downloadReportPdf({ prediction, report, image, patientName });
      toast.success("PDF downloaded");
    } catch (err) {
      toast.error(`Couldn't create the PDF: ${err.message}`);
    } finally {
      setExporting(false);
    }
  }

  const templateOnly = health.data && !health.data.llm_configured;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Sparkles className="size-4 text-primary" aria-hidden="true" /> Plain-language report
        </CardTitle>
        <CardDescription>
          An AI explains the classifier&apos;s result in plain words. It can&apos;t change the result: the class, the
          probability and the triage status are checked against the classifier before you see them.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <AnimatePresence mode="wait">
          {reporting ? (
            <ReportProgress
              key="loading"
              status={reportStatus}
              startedAt={reportStartedAt}
              inline={health.data?.report_mode === "inline"}
              workers={health.data?.report_workers}
            />
          ) : report ? (
            <ReportBody key="report" result={report} />
          ) : reportError ? (
            <div key="error" role="alert" className="space-y-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm">
              <p className="flex items-center gap-2 font-medium">
                <TriangleAlert className="size-4 shrink-0 text-destructive" aria-hidden="true" />
                Couldn&apos;t write the report
              </p>
              <p>{reportError.message}</p>
              <p className="text-xs text-muted-foreground">
                The classifier&apos;s result above is unaffected, and the PDF works without the report.
              </p>
              <Button size="sm" variant="outline" onClick={generateReport}>
                <RotateCw aria-hidden="true" /> Try again
              </Button>
            </div>
          ) : (
            <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-start gap-3">
              <Button onClick={generateReport} disabled={!prediction}>
                <FileText aria-hidden="true" /> Generate report
              </Button>
              {templateOnly && (
                <p className="text-xs text-muted-foreground">
                  No LLM key is set on the server, so you&apos;ll get the standard template report.
                </p>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid gap-3 border-t pt-5 sm:grid-cols-[1fr_auto] sm:items-end">
          <div className="space-y-2">
            <Label htmlFor="patient-name">Patient name for the PDF (optional)</Label>
            <Input
              id="patient-name"
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
              placeholder="Stays on this device"
              autoComplete="off"
              aria-describedby="patient-hint"
            />
            <p id="patient-hint" className="text-xs text-muted-foreground">
              Never sent to the server; it only goes into the downloaded file.
            </p>
          </div>
          <Button variant="outline" onClick={downloadPdf} disabled={!prediction || exporting} className="sm:mb-6">
            {exporting ? <Loader2 className="animate-spin" aria-hidden="true" /> : <Download aria-hidden="true" />}
            Download PDF
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
