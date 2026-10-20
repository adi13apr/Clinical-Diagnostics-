import { AnimatePresence, motion } from "framer-motion";
import { Download, FileText, Info, Loader2, RotateCw, Sparkles } from "lucide-react";
import { useState } from "react";
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

function ReportSkeleton() {
  return (
    <div className="space-y-4" role="status" aria-label="Writing the report">
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" aria-hidden="true" />
        Writing the report. On the free LLM tier this can take up to a minute; you can visit other pages meanwhile.
      </p>
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
  const { prediction, report, reporting, reportError, generateReport, previewUrl, opacity, patientName, setPatientName } =
    useDiagnosis();
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
            <ReportSkeleton key="loading" />
          ) : report ? (
            <ReportBody key="report" result={report} />
          ) : reportError ? (
            <div key="error" role="alert" className="space-y-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm">
              <p>Couldn&apos;t write the report: {reportError.message}</p>
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
