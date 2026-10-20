import { BarChart3, Table2 } from "lucide-react";
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

/** Card with a Chart / Table switch, so no value is only readable from colour or hover. */
export function ChartCard({ title, description, chart, table, legend, className }) {
  const [view, setView] = useState("chart");
  return (
    <Card className={className}>
      <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
        <div className="min-w-0 space-y-1">
          <CardTitle className="text-base">{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </div>
        <div role="group" aria-label={`${title}: view`} className="flex shrink-0 rounded-lg border p-0.5">
          {[
            ["chart", BarChart3, "Chart"],
            ["table", Table2, "Table"],
          ].map(([value, Icon, label]) => (
            <button
              key={value}
              type="button"
              aria-pressed={view === value}
              onClick={() => setView(value)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
                view === value ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Icon className="size-3.5" aria-hidden="true" />
              <span className="sr-only sm:not-sr-only">{label}</span>
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        {view === "chart" ? (
          <>
            {legend}
            <div aria-hidden="true">{chart}</div>
            <p className="sr-only">Switch to the table view to read the exact values.</p>
          </>
        ) : (
          <div className="max-h-80 overflow-auto rounded-lg border">{table}</div>
        )}
      </CardContent>
    </Card>
  );
}

/** Legend row: a colour swatch beside ink-coloured text. */
export function Legend({ items }) {
  return (
    <ul className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {items.map(({ label, color }) => (
        <li key={label} className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-sm" style={{ background: color }} aria-hidden="true" />
          {label}
        </li>
      ))}
    </ul>
  );
}

/** Tooltip box shared by the charts. */
export function ChartTooltip({ active, payload, label, formatLabel, formatValue }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-popover px-3 py-2 text-xs shadow-md">
      <p className="mb-1 font-medium text-popover-foreground">{formatLabel ? formatLabel(label) : label}</p>
      <ul className="space-y-0.5">
        {payload.map((p) => (
          <li key={p.dataKey} className="flex items-center justify-between gap-4">
            <span className="inline-flex items-center gap-1.5 text-muted-foreground">
              <span className="size-2 rounded-sm" style={{ background: p.color }} aria-hidden="true" />
              {p.name}
            </span>
            <span className="tabular font-medium text-popover-foreground">
              {formatValue ? formatValue(p.value, p) : p.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
