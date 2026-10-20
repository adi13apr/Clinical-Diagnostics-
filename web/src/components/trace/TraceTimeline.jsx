import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, ChevronRight, Diamond, XCircle } from "lucide-react";
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { isEvent } from "@/components/trace/traceModel";
import { duration, formatDateTime, STEP_LABELS } from "@/lib/format";
import { cn } from "@/lib/utils";

const GROUP_TITLES = { prediction: "Prediction", gradcam: "Grad-CAM", report: "AI report" };

function formatValue(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(v < 0.01 ? 6 : 3);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function SpanDetails({ span }) {
  const attrs = Object.entries(span.attributes ?? {});
  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: "auto", opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="overflow-hidden"
    >
      <div className="mx-2 mb-2 rounded-lg border bg-muted/40 p-3 text-xs">
        {span.status === "error" && (
          <p className="mb-2 text-destructive">
            <span className="font-semibold">{span.error_type || "Error"}</span>
            {span.error && `: ${span.error}`}
          </p>
        )}
        <dl className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">Started</dt>
            <dd className="tabular">{formatDateTime(span.start)}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">Duration</dt>
            <dd className="tabular">{isEvent(span) ? "event" : duration(span.latency_ms)}</dd>
          </div>
          {attrs.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-3">
              <dt className="text-muted-foreground">{k}</dt>
              <dd className="tabular truncate text-right" title={formatValue(v)}>
                {formatValue(v)}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </motion.div>
  );
}

function SpanRow({ row, open, onToggle, index }) {
  const { span, depth, offset, width } = row;
  const error = span.status === "error";
  const event = isEvent(span);
  const label = STEP_LABELS[span.name] ?? span.name;
  let extra = "";
  if (span.name === "guard" && span.attributes?.result) extra = ` · ${span.attributes.result}`;
  else if (span.name === "fallback_used") extra = span.attributes?.used ? " · used" : " · not needed";
  else if (span.name === "llm_call" && span.attributes?.step) extra = ` · step ${span.attributes.step}`;
  else if (span.name === "tool_call" && span.attributes?.tool) extra = ` · ${span.attributes.tool}`;
  const detailsId = `span-${span.span_id}`;

  return (
    <li>
      <motion.button
        type="button"
        initial={{ opacity: 0, x: -6 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.2, delay: Math.min(index * 0.03, 0.4) }}
        onClick={onToggle}
        aria-expanded={open}
        aria-controls={detailsId}
        className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 rounded-lg px-2 py-2 text-left text-sm outline-none hover:bg-muted/50 focus-visible:ring-[3px] focus-visible:ring-ring/50 md:grid-cols-[14rem_minmax(0,1fr)_5rem]"
      >
        <span className="flex min-w-0 items-center gap-1.5" style={{ paddingLeft: `${depth * 0.9}rem` }}>
          <ChevronRight className={cn("size-3.5 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")} aria-hidden="true" />
          {error ? (
            <XCircle className="size-4 shrink-0 text-destructive" aria-label="Error" />
          ) : (
            <CheckCircle2 className="size-4 shrink-0 text-success" aria-label="OK" />
          )}
          <span className="truncate">
            {label}
            <span className="text-muted-foreground">{extra}</span>
          </span>
        </span>
        {/* Waterfall bar: position = when it started, width = how long it took */}
        <span className="relative hidden h-5 md:block" aria-hidden="true">
          <span className="absolute inset-y-[9px] right-0 left-0 rounded-full bg-muted" />
          {event ? (
            <Diamond
              className={cn("absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2", error ? "text-destructive" : "text-primary")}
              fill="currentColor"
              style={{ left: `${offset * 100}%` }}
            />
          ) : (
            <motion.span
              className={cn("absolute inset-y-1 rounded", error ? "bg-destructive" : depth === 0 ? "bg-primary" : "bg-primary/60")}
              style={{ left: `${offset * 100}%` }}
              initial={{ width: 0 }}
              animate={{ width: `max(3px, ${width * 100}%)` }}
              transition={{ duration: 0.5, delay: Math.min(index * 0.03, 0.4), ease: "easeOut" }}
            />
          )}
        </span>
        <span className="tabular text-right text-xs text-muted-foreground">{event ? "event" : duration(span.latency_ms)}</span>
      </motion.button>
      <div id={detailsId}>
        <AnimatePresence initial={false}>{open && <SpanDetails span={span} />}</AnimatePresence>
      </div>
    </li>
  );
}

/** One card per request in the trace, each a waterfall of its steps. */
export function TraceTimeline({ groups }) {
  const [open, setOpen] = useState(() => new Set());
  const toggle = (id) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="space-y-6">
      {groups.map((g) => (
        <Card key={g.root.span_id}>
          <CardHeader>
            <CardTitle className="flex flex-wrap items-center justify-between gap-2 text-base">
              <span>{GROUP_TITLES[g.root.name] ?? STEP_LABELS[g.root.name] ?? g.root.name}</span>
              <span className="tabular text-sm font-normal text-muted-foreground">{duration(g.totalMs)}</span>
            </CardTitle>
            <CardDescription>{formatDateTime(g.root.start)}</CardDescription>
          </CardHeader>
          <CardContent className="px-3 sm:px-4">
            <div className="hidden px-2 pb-1 text-xs text-muted-foreground md:grid md:grid-cols-[14rem_minmax(0,1fr)_5rem] md:gap-3">
              <span>Step</span>
              <span className="flex justify-between">
                <span>0</span>
                <span>{duration(g.totalMs)}</span>
              </span>
              <span className="text-right">Time</span>
            </div>
            <ul className="space-y-0.5">
              {g.rows.map((row, i) => (
                <SpanRow
                  key={row.span.span_id}
                  row={row}
                  index={i}
                  open={open.has(row.span.span_id)}
                  onToggle={() => toggle(row.span.span_id)}
                />
              ))}
            </ul>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
