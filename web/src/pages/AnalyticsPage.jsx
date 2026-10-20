import { Activity, BarChart3, Clock, Coins, ShieldCheck, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import { ReferralChart, StepLatencyChart, VolumeChart } from "@/components/analytics/Charts";
import { StatTile, StatTilesSkeleton } from "@/components/analytics/StatTiles";
import { PageHeader } from "@/components/common/PageHeader";
import { EmptyState, ErrorState } from "@/components/common/States";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useAnalytics } from "@/hooks/queries";
import { compact, duration, percent } from "@/lib/format";
import { cn } from "@/lib/utils";

const RANGES = [7, 30, 90];

function RangePicker({ days, onChange }) {
  return (
    <div role="group" aria-label="Time range" className="flex rounded-lg border bg-card p-0.5">
      {RANGES.map((d) => (
        <button
          key={d}
          type="button"
          aria-pressed={days === d}
          onClick={() => onChange(d)}
          className={cn(
            "rounded-md px-3 py-1.5 text-sm font-medium outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
            days === d ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {d} days
        </button>
      ))}
    </div>
  );
}

function weightedLatency(daily) {
  let total = 0;
  let n = 0;
  for (const d of daily) {
    if (d.avg_inference_ms !== null) {
      total += d.avg_inference_ms * d.predictions;
      n += d.predictions;
    }
  }
  return n ? total / n : null;
}

function PipelineCard({ pipeline }) {
  const rows = [
    ["Traces analysed", compact(pipeline.traces)],
    ["Reports written by the LLM", compact(pipeline.reports_with_llm)],
    ["Input tokens per report", compact(pipeline.avg_input_tokens_per_report)],
    ["Output tokens per report", compact(pipeline.avg_output_tokens_per_report)],
    [
      "Nominal cost per report",
      pipeline.avg_nominal_cost_per_report_usd === null ? "—" : `$${pipeline.avg_nominal_cost_per_report_usd.toFixed(4)}`,
    ],
    [
      "Guard rejections",
      pipeline.guard_checks ? `${percent(pipeline.guard_reject_rate)} of ${pipeline.guard_checks} checks` : "No LLM drafts yet",
    ],
    [
      "Template fallbacks",
      pipeline.reports ? `${pipeline.fallback_rate === 1 ? "All" : percent(pipeline.fallback_rate)} of ${pipeline.reports} report${pipeline.reports === 1 ? "" : "s"}` : "No reports yet",
    ],
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <ShieldCheck className="size-4 text-primary" aria-hidden="true" /> Report pipeline
        </CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="divide-y text-sm">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between gap-4 py-2.5">
              <dt className="text-muted-foreground">{k}</dt>
              <dd className="tabular text-right font-medium">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-xs text-muted-foreground">
          Cost is nominal: the free tier bills $0. A guard rejection means a draft disagreed with the classifier and was
          rewritten or replaced.
        </p>
      </CardContent>
    </Card>
  );
}

export function AnalyticsPage() {
  const [days, setDays] = useState(30);
  const query = useAnalytics(days);
  const data = query.data;

  let body;
  if (query.isPending) {
    body = (
      <div className="space-y-6" role="status" aria-label="Loading analytics">
        <StatTilesSkeleton />
        <div className="grid gap-6 lg:grid-cols-2">
          <Skeleton className="h-80 rounded-xl" />
          <Skeleton className="h-80 rounded-xl" />
        </div>
      </div>
    );
  } else if (query.isError) {
    body = <ErrorState title="Couldn't load analytics" error={query.error} onRetry={query.refetch} />;
  } else if (data.total_predictions === 0) {
    body = (
      <EmptyState
        icon={BarChart3}
        title={`No predictions in the last ${days} days`}
        description="Charts appear once you've analysed some X-rays."
        action={
          <Button asChild>
            <Link to="/">Analyse an X-ray</Link>
          </Button>
        }
      />
    );
  } else {
    const avgMs = weightedLatency(data.daily);
    body = (
      <div className="space-y-6">
        {data.warnings.length > 0 && (
          <Alert>
            <TriangleAlert aria-hidden="true" />
            <AlertTitle>Some data couldn&apos;t be loaded</AlertTitle>
            <AlertDescription>
              <ul className="list-disc pl-4">
                {data.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <StatTile label="Predictions" value={compact(data.total_predictions)} detail={`Last ${days} days`} icon={Activity} />
          <StatTile
            label="Referred"
            value={percent(data.referral_rate)}
            detail={`${data.total_referred} sent for clinician review`}
            icon={TriangleAlert}
            delay={0.05}
          />
          <StatTile label="Model time" value={duration(avgMs)} detail="Average per X-ray" icon={Clock} delay={0.1} />
          <StatTile
            label="Cost per report"
            value={
              data.pipeline?.avg_nominal_cost_per_report_usd == null
                ? "—"
                : `$${data.pipeline.avg_nominal_cost_per_report_usd.toFixed(3)}`
            }
            detail="Nominal; billed $0 on the free tier"
            icon={Coins}
            delay={0.15}
          />
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          <VolumeChart daily={data.daily} />
          <ReferralChart daily={data.daily} />
        </div>
        {data.pipeline ? (
          <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
            {Object.keys(data.pipeline.steps).length > 0 ? (
              <StepLatencyChart steps={data.pipeline.steps} />
            ) : (
              <EmptyState icon={Clock} title="No traces yet" description="Step timings appear after your next prediction." />
            )}
            <PipelineCard pipeline={data.pipeline} />
          </div>
        ) : (
          <EmptyState icon={Clock} title="Pipeline metrics unavailable" description="Traces couldn't be loaded right now." />
        )}
      </div>
    );
  }

  return (
    <>
      <PageHeader
        title="Analytics"
        description="Your prediction volume, referral rate and how long each step takes."
        actions={<RangePicker days={days} onChange={setDays} />}
      />
      {body}
    </>
  );
}
