import { FileText, ImageUp, RotateCcw, ScanSearch } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/common/PageHeader";
import { ErrorState } from "@/components/common/States";
import { GradcamViewer } from "@/components/diagnose/GradcamViewer";
import { ReportPanel } from "@/components/diagnose/ReportPanel";
import { ResultCard, ResultCardSkeleton } from "@/components/diagnose/ResultCard";
import { UploadDropzone } from "@/components/diagnose/UploadDropzone";
import { Button } from "@/components/ui/button";
import { useDiagnosis } from "@/providers/contexts";

const STEPS = [
  { icon: ImageUp, title: "Upload", text: "A frontal chest X-ray, JPEG or PNG. It stays in memory for an hour, never on disk." },
  { icon: ScanSearch, title: "Review", text: "Calibrated confidence, a Confident or Refer call, and a heatmap of where the model looked." },
  { icon: FileText, title: "Report", text: "Generate a plain-language report and download it as a PDF." },
];

function HowItWorks() {
  return (
    <ol className="mt-6 grid gap-4 sm:grid-cols-3">
      {STEPS.map(({ icon: Icon, title, text }, i) => (
        <li key={title} className="flex gap-3 rounded-xl border bg-card/60 p-4">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-foreground">
            <Icon className="size-4" aria-hidden="true" />
          </span>
          <span>
            <span className="block text-sm font-semibold">
              {i + 1}. {title}
            </span>
            <span className="mt-0.5 block text-sm text-muted-foreground">{text}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}

export function DiagnosePage() {
  const { file, previewUrl, prediction, predicting, predictError, opacity, setOpacity, analyze, retry, reset } =
    useDiagnosis();

  const onError = (message) => toast.error(message);
  const hasUpload = Boolean(file);

  return (
    <>
      <PageHeader
        title="Diagnose"
        description="Upload a chest X-ray to see the model's most likely class, how sure it is, and where it looked."
        actions={
          hasUpload && (
            <Button variant="outline" onClick={reset} disabled={predicting}>
              <RotateCcw aria-hidden="true" /> New image
            </Button>
          )
        }
      />

      {!hasUpload ? (
        <>
          <UploadDropzone onFile={analyze} onError={onError} />
          <HowItWorks />
        </>
      ) : (
        <div className="space-y-6">
          <div className="grid items-start gap-6 lg:grid-cols-2">
            <GradcamViewer
              previewUrl={previewUrl}
              traceId={prediction?.trace_id}
              opacity={opacity}
              onOpacityChange={setOpacity}
              scanning={predicting}
            />
            <div className="space-y-6">
              {predicting && <ResultCardSkeleton />}
              {predictError && !predicting && (
                <ErrorState title="The X-ray couldn't be analysed" error={predictError} onRetry={retry} />
              )}
              {prediction && !predicting && <ResultCard prediction={prediction} />}
              {!predicting && <UploadDropzone compact onFile={analyze} onError={onError} />}
            </div>
          </div>
          {prediction && !predicting && <ReportPanel />}
        </div>
      )}
    </>
  );
}
