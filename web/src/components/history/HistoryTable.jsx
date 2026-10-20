import { motion } from "framer-motion";
import { ChevronRight } from "lucide-react";
import { Link } from "react-router";
import { ClassLabel, StatusBadge } from "@/components/common/Badges";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { CLASS_COLORS, duration, formatDateTime, probability, shortId } from "@/lib/format";

function Confidence({ item }) {
  const raw = item.temperature === null || item.temperature === undefined;
  return (
    <span className="inline-flex items-center gap-2">
      <span className="tabular font-medium">{probability(item.confidence)}</span>
      <span className="hidden h-1.5 w-14 overflow-hidden rounded-full bg-muted sm:inline-block" aria-hidden="true">
        <span
          className="block h-full rounded-full"
          style={{ width: `${item.confidence * 100}%`, background: CLASS_COLORS[item.predicted_label] }}
        />
      </span>
      {raw && (
        <Tooltip>
          <TooltipTrigger asChild>
            <button type="button" className="rounded text-xs text-muted-foreground underline decoration-dotted outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">
              raw
            </button>
          </TooltipTrigger>
          <TooltipContent>Logged before calibration: this is the raw, much lower probability.</TooltipContent>
        </Tooltip>
      )}
    </span>
  );
}

function TraceLink({ id, children, className }) {
  if (!id) return <span className="text-muted-foreground">—</span>;
  return (
    <Link
      to={`/traces/${id}`}
      title={id}
      className={`rounded font-mono text-xs text-primary underline-offset-4 outline-none hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/50 ${className ?? ""}`}
    >
      {children ?? shortId(id)}
    </Link>
  );
}

/** Table on tablet/desktop, stacked cards on phones. */
export function HistoryTable({ items }) {
  return (
    <>
      <div className="hidden overflow-hidden rounded-xl border bg-card md:block">
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/50 hover:bg-muted/50">
              <TableHead className="pl-4">Date</TableHead>
              <TableHead>Result</TableHead>
              <TableHead>Confidence</TableHead>
              <TableHead>Triage</TableHead>
              <TableHead className="text-right">Time</TableHead>
              <TableHead className="pr-4 text-right">Trace</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((item, i) => (
              <motion.tr
                key={item.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2, delay: Math.min(i * 0.02, 0.3) }}
                className="border-b transition-colors hover:bg-muted/40"
              >
                <TableCell className="pl-4 whitespace-nowrap">{formatDateTime(item.created_at)}</TableCell>
                <TableCell>
                  <ClassLabel name={item.predicted_label} />
                </TableCell>
                <TableCell>
                  <Confidence item={item} />
                </TableCell>
                <TableCell>
                  <StatusBadge status={item.status} />
                </TableCell>
                <TableCell className="tabular text-right text-muted-foreground">{duration(item.inference_ms)}</TableCell>
                <TableCell className="pr-4 text-right">
                  <TraceLink id={item.trace_id} />
                </TableCell>
              </motion.tr>
            ))}
          </TableBody>
        </Table>
      </div>

      <ul className="space-y-3 md:hidden">
        {items.map((item) => (
          <li key={item.id} className="rounded-xl border bg-card p-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 space-y-1">
                <ClassLabel name={item.predicted_label} className="font-medium" />
                <p className="text-xs text-muted-foreground">{formatDateTime(item.created_at)}</p>
              </div>
              <StatusBadge status={item.status} />
            </div>
            <div className="mt-3 flex items-center justify-between text-sm">
              <Confidence item={item} />
              {item.trace_id && (
                <TraceLink id={item.trace_id} className="inline-flex items-center gap-0.5 font-sans">
                  Trace <ChevronRight className="size-3.5" aria-hidden="true" />
                </TraceLink>
              )}
            </div>
          </li>
        ))}
      </ul>
    </>
  );
}

export function HistoryTableSkeleton() {
  return (
    <div className="space-y-2 rounded-xl border bg-card p-4" role="status" aria-label="Loading history">
      {Array.from({ length: 6 }, (_, i) => (
        <div key={i} className="flex items-center gap-4 py-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-20" />
          <Skeleton className="ml-auto h-5 w-20 rounded-full" />
        </div>
      ))}
    </div>
  );
}
