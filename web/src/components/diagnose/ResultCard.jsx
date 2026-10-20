import { motion } from "framer-motion";
import { Activity, Info } from "lucide-react";
import { Link } from "react-router";
import { ClassLabel, StatusBadge } from "@/components/common/Badges";
import { ConfidenceGauge } from "@/components/diagnose/ConfidenceGauge";
import { ProbabilityBars } from "@/components/diagnose/ProbabilityBars";
import { TriageBanner } from "@/components/diagnose/TriageBanner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { CLASS_BLURBS, duration } from "@/lib/format";

/** The prediction: class, gauge, triage banner, per-class probabilities, and run details. */
export function ResultCard({ prediction }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <Card aria-live="polite">
        <CardHeader className="gap-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Model&apos;s most likely class</span>
            <StatusBadge status={prediction.status} />
          </div>
          <CardTitle className="text-2xl sm:text-3xl">
            <ClassLabel name={prediction.predicted_class} className="gap-3 [&>span:first-child]:size-3.5" />
          </CardTitle>
          <p className="text-sm text-muted-foreground">{CLASS_BLURBS[prediction.predicted_class]}</p>
        </CardHeader>
        <CardContent className="space-y-6">
          <ConfidenceGauge
            value={prediction.confidence}
            threshold={prediction.referral_threshold}
            status={prediction.status}
          />
          <TriageBanner prediction={prediction} />
          <section aria-labelledby="probs-title">
            <h3 id="probs-title" className="mb-3 text-sm font-semibold">
              Calibrated probability per class
            </h3>
            <ProbabilityBars probabilities={prediction.probabilities} predicted={prediction.predicted_class} />
          </section>
          <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-4 text-xs text-muted-foreground">
            <span className="tabular">
              Analysed in {duration(prediction.inference_ms)} · T = {prediction.temperature.toFixed(3)}
            </span>
            <Link
              to={`/traces/${prediction.trace_id}`}
              className="inline-flex items-center gap-1 rounded font-medium text-primary underline-offset-4 outline-none hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/50"
            >
              <Activity className="size-3.5" aria-hidden="true" /> View trace
            </Link>
          </div>
          {prediction.warnings?.length > 0 && (
            <ul className="space-y-1 text-xs text-muted-foreground">
              {prediction.warnings.map((w) => (
                <li key={w} className="flex gap-1.5">
                  <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" /> {w}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function ResultCardSkeleton() {
  return (
    <Card aria-busy="true" aria-label="Analysing the X-ray">
      <CardHeader className="gap-3">
        <Skeleton className="h-3 w-40" />
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-4 w-64" />
      </CardHeader>
      <CardContent className="space-y-6">
        <Skeleton className="mx-auto h-32 w-56 rounded-t-full" />
        <Skeleton className="h-16 w-full rounded-xl" />
        <div className="space-y-4">
          {[0, 1, 2].map((i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-2 w-full" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
