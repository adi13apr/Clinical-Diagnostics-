import { Activity, ArrowLeft, ArrowRight, Copy, SearchX } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { toast } from "sonner";
import { StatTile } from "@/components/analytics/StatTiles";
import { ClassLabel, StatusBadge } from "@/components/common/Badges";
import { PageHeader } from "@/components/common/PageHeader";
import { EmptyState, ErrorState } from "@/components/common/States";
import { TraceTimeline } from "@/components/trace/TraceTimeline";
import { buildGroups, summarize } from "@/components/trace/traceModel";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { useHistory, useTrace } from "@/hooks/queries";
import { compact, formatDateTime, percent, shortId } from "@/lib/format";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** /traces: open a trace by id, or pick one of your recent predictions. */
function TracePicker() {
  const navigate = useNavigate();
  const [value, setValue] = useState("");
  const [error, setError] = useState(null);
  const recent = useHistory({ limit: 8, offset: 0 });
  const items = (recent.data?.items ?? []).filter((it) => it.trace_id);

  function submit(e) {
    e.preventDefault();
    const id = value.trim();
    if (!UUID.test(id)) {
      setError("A trace ID looks like 3f2b1c4d-0000-4000-8000-1234567890ab.");
      return;
    }
    navigate(`/traces/${id}`);
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Open a trace</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-3">
            <Label htmlFor="trace-id">Trace ID</Label>
            <div className="flex gap-2">
              <Input
                id="trace-id"
                value={value}
                onChange={(e) => {
                  setValue(e.target.value);
                  setError(null);
                }}
                placeholder="Paste a trace ID"
                className="font-mono text-sm"
                aria-invalid={Boolean(error)}
                aria-describedby={error ? "trace-id-error" : undefined}
              />
              <Button type="submit" aria-label="Open trace">
                <ArrowRight aria-hidden="true" />
              </Button>
            </div>
            {error && (
              <p id="trace-id-error" role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
          </form>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recent predictions</CardTitle>
        </CardHeader>
        <CardContent>
          {recent.isPending ? (
            <div className="space-y-3" role="status" aria-label="Loading recent predictions">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : recent.isError ? (
            <ErrorState error={recent.error} onRetry={recent.refetch} className="py-6" />
          ) : items.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No traced predictions yet. <Link to="/" className="text-primary underline-offset-4 hover:underline">Analyse an X-ray</Link> to
              create one.
            </p>
          ) : (
            <ul className="divide-y">
              {items.map((it) => (
                <li key={it.id}>
                  <Link
                    to={`/traces/${it.trace_id}`}
                    className="flex items-center justify-between gap-3 rounded-md px-2 py-2.5 text-sm outline-none hover:bg-muted/50 focus-visible:ring-[3px] focus-visible:ring-ring/50"
                  >
                    <span className="min-w-0">
                      <ClassLabel name={it.predicted_label} className="font-medium" />
                      <span className="block text-xs text-muted-foreground">
                        {formatDateTime(it.created_at)} · <span className="font-mono">{shortId(it.trace_id)}</span>
                      </span>
                    </span>
                    <span className="flex items-center gap-2">
                      <StatusBadge status={it.status} />
                      <ArrowRight className="size-4 text-muted-foreground" aria-hidden="true" />
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function TraceDetail({ traceId }) {
  const query = useTrace(traceId);
  const spans = query.data?.spans;
  const groups = useMemo(() => (spans ? buildGroups(spans) : []), [spans]);
  const totals = useMemo(() => (spans ? summarize(spans) : null), [spans]);

  if (query.isPending) {
    return (
      <div className="space-y-6" role="status" aria-label="Loading trace">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28 rounded-xl" />
          ))}
        </div>
        <Skeleton className="h-72 rounded-xl" />
      </div>
    );
  }
  if (query.isError) {
    if (query.error.status === 404) {
      return (
        <EmptyState
          icon={SearchX}
          title="Trace not found"
          description="It may belong to another account, or it was never saved to the database."
          action={
            <Button asChild variant="outline">
              <Link to="/traces">Back to traces</Link>
            </Button>
          }
        />
      );
    }
    return <ErrorState title="Couldn't load this trace" error={query.error} onRetry={query.refetch} />;
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile label="Steps recorded" value={compact(totals.spans)} detail={`${groups.length} request${groups.length === 1 ? "" : "s"}`} />
        <StatTile
          label="Errors"
          value={compact(totals.errors)}
          detail={totals.errors ? "Open a red step for details" : "Every step succeeded"}
          delay={0.05}
        />
        <StatTile label="LLM calls" value={compact(totals.llmCalls)} detail={`${compact(totals.tokens)} tokens`} delay={0.1} />
        <StatTile
          label="Nominal cost"
          value={totals.llmCalls ? `$${totals.cost.toFixed(4)}` : "—"}
          detail={totals.errors ? `${percent(totals.errors / totals.spans)} of steps failed` : "Billed $0 on the free tier"}
          delay={0.15}
        />
      </div>
      <TraceTimeline groups={groups} />
    </div>
  );
}

export function TracePage() {
  const { traceId } = useParams();
  const copy = () =>
    navigator.clipboard
      .writeText(traceId)
      .then(() => toast.success("Trace ID copied"))
      .catch(() => toast.error("Couldn't copy"));

  return (
    <>
      <PageHeader
        title={traceId ? "Trace" : "Traces"}
        description={
          traceId ? (
            <span className="font-mono text-sm break-all">{traceId}</span>
          ) : (
            "Every step of a prediction: what ran, whether it worked, and how long it took."
          )
        }
        actions={
          traceId && (
            <>
              <Button variant="ghost" asChild>
                <Link to="/traces">
                  <ArrowLeft aria-hidden="true" /> All traces
                </Link>
              </Button>
              <Button variant="outline" onClick={copy}>
                <Copy aria-hidden="true" /> Copy ID
              </Button>
            </>
          )
        }
      />
      {traceId ? (
        UUID.test(traceId) ? (
          <TraceDetail traceId={traceId} />
        ) : (
          <EmptyState icon={Activity} title="That isn't a trace ID" description="Trace IDs are UUIDs, like the ones on the History page." />
        )
      ) : (
        <TracePicker />
      )}
    </>
  );
}
