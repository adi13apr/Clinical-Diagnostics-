import { Search, SlidersHorizontal, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CLASSES } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * Filter row. label/status/dates go to the server; `q` filters the loaded page in the browser
 * (the API has no text search, and predictions carry no patient names).
 */
export function HistoryFilters({ filters, onChange, onClear }) {
  const active = filters.label || filters.status || filters.dateFrom || filters.dateTo || filters.q;
  const set = (key) => (value) => onChange({ [key]: value === "all" ? "" : value });
  const serverFilters = [filters.label, filters.status, filters.dateFrom, filters.dateTo].filter(Boolean).length;
  const [open, setOpen] = useState(serverFilters > 0);
  // On phones the filters fold away behind a button; from sm up they're always shown
  const fold = cn(!open && "hidden", "sm:block");

  return (
    <div className="grid gap-3 rounded-xl border bg-card p-4 sm:grid-cols-2 lg:grid-cols-[1.4fr_1fr_1fr_1fr_1fr_auto] lg:items-end">
      <div className="space-y-1.5 sm:col-span-2 lg:col-span-1">
        <Label htmlFor="history-search">Search this page</Label>
        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
          <Input
            id="history-search"
            type="search"
            value={filters.q}
            onChange={(e) => onChange({ q: e.target.value }, { replace: true })}
            placeholder="Trace ID, class or date"
            className="pl-9"
          />
        </div>
        <Button
          variant="outline"
          size="sm"
          className="mt-1 sm:hidden"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls="history-more-filters"
        >
          <SlidersHorizontal aria-hidden="true" /> Filters{serverFilters > 0 && ` (${serverFilters})`}
        </Button>
      </div>
      <div className={cn("space-y-1.5", fold)} id="history-more-filters">
        <Label htmlFor="history-label">Result</Label>
        <Select value={filters.label || "all"} onValueChange={set("label")}>
          <SelectTrigger id="history-label" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All classes</SelectItem>
            {CLASSES.map((c) => (
              <SelectItem key={c} value={c}>
                {c}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className={cn("space-y-1.5", fold)}>
        <Label htmlFor="history-status">Triage</Label>
        <Select value={filters.status || "all"} onValueChange={set("status")}>
          <SelectTrigger id="history-status" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All</SelectItem>
            <SelectItem value="confident">Confident</SelectItem>
            <SelectItem value="refer">Refer</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className={cn("space-y-1.5", fold)}>
        <Label htmlFor="history-from">From (UTC)</Label>
        <Input
          id="history-from"
          type="date"
          value={filters.dateFrom}
          max={filters.dateTo || undefined}
          onChange={(e) => onChange({ dateFrom: e.target.value })}
        />
      </div>
      <div className={cn("space-y-1.5", fold)}>
        <Label htmlFor="history-to">To (UTC)</Label>
        <Input
          id="history-to"
          type="date"
          value={filters.dateTo}
          min={filters.dateFrom || undefined}
          onChange={(e) => onChange({ dateTo: e.target.value })}
        />
      </div>
      <Button variant="ghost" onClick={onClear} disabled={!active} className={cn("justify-self-start lg:justify-self-auto", !open && "hidden", "sm:inline-flex")}>
        <X aria-hidden="true" /> Clear
      </Button>
    </div>
  );
}
