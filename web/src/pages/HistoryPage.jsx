import { ChevronLeft, ChevronRight, History, Loader2, SearchX } from "lucide-react";
import { useMemo } from "react";
import { Link, useSearchParams } from "react-router";
import { PageHeader } from "@/components/common/PageHeader";
import { EmptyState, ErrorState } from "@/components/common/States";
import { HistoryFilters } from "@/components/history/HistoryFilters";
import { HistoryTable, HistoryTableSkeleton } from "@/components/history/HistoryTable";
import { Button } from "@/components/ui/button";
import { useHistory } from "@/hooks/queries";
import { formatDateTime } from "@/lib/format";

const PAGE_SIZE = 20;
const KEYS = ["q", "label", "status", "dateFrom", "dateTo"];

function matches(item, q) {
  const needle = q.trim().toLowerCase();
  if (!needle) return true;
  return [item.trace_id, item.predicted_label, item.status, formatDateTime(item.created_at), item.created_at]
    .filter(Boolean)
    .some((v) => String(v).toLowerCase().includes(needle));
}

export function HistoryPage() {
  // Filters live in the URL: shareable, and Back restores them
  const [params, setParams] = useSearchParams();
  const filters = Object.fromEntries(KEYS.map((k) => [k, params.get(k) ?? ""]));
  const page = Math.max(0, Number(params.get("page") ?? 0) || 0);

  const query = useHistory({
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
    label: filters.label || undefined,
    status: filters.status || undefined,
    dateFrom: filters.dateFrom || undefined,
    dateTo: filters.dateTo || undefined,
  });

  function update(changes, { replace = false } = {}) {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        for (const [k, v] of Object.entries(changes)) {
          if (v) next.set(k, v);
          else next.delete(k);
        }
        if (!("page" in changes) && !("q" in changes)) next.delete("page"); // new server filter -> page 1
        return next;
      },
      { replace },
    );
  }

  const items = useMemo(() => (query.data?.items ?? []).filter((it) => matches(it, filters.q)), [query.data, filters.q]);
  const total = query.data?.total ?? 0;
  const first = total ? page * PAGE_SIZE + 1 : 0;
  const last = Math.min(total, (page + 1) * PAGE_SIZE);
  const serverFiltered = filters.label || filters.status || filters.dateFrom || filters.dateTo;
  const clear = () => setParams({});

  let body;
  if (query.isPending) body = <HistoryTableSkeleton />;
  else if (query.isError) body = <ErrorState title="Couldn't load your history" error={query.error} onRetry={query.refetch} />;
  else if (total === 0 && !serverFiltered)
    body = (
      <EmptyState
        icon={History}
        title="No predictions yet"
        description="Every X-ray you analyse appears here, with its result, confidence and triage status."
        action={
          <Button asChild>
            <Link to="/">Analyse an X-ray</Link>
          </Button>
        }
      />
    );
  else if (items.length === 0)
    body = (
      <EmptyState
        icon={SearchX}
        title="No matching predictions"
        description={filters.q ? "Nothing on this page matches your search." : "Try a wider date range or another filter."}
        action={
          <Button variant="outline" onClick={clear}>
            Clear filters
          </Button>
        }
      />
    );
  else body = <HistoryTable items={items} />;

  return (
    <>
      <PageHeader
        title="History"
        description="Your past predictions, newest first. Patient names are never stored with them."
      />
      <div className="space-y-4">
        <HistoryFilters filters={filters} onChange={update} onClear={clear} />
        <div className="relative">
          {query.isFetching && !query.isPending && (
            <Loader2 className="absolute -top-7 right-0 size-4 animate-spin text-muted-foreground" aria-label="Updating" />
          )}
          {body}
        </div>
        {total > 0 && !query.isError && (
          <nav aria-label="Pagination" className="flex flex-wrap items-center justify-between gap-3 text-sm">
            <p className="tabular text-muted-foreground" aria-live="polite">
              {first}–{last} of {total}
              {filters.q && ` · ${items.length} match${items.length === 1 ? "" : "es"} on this page`}
            </p>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" disabled={page === 0} onClick={() => update({ page: page - 1 ? String(page - 1) : "" })}>
                <ChevronLeft aria-hidden="true" /> Previous
              </Button>
              <Button variant="outline" size="sm" disabled={last >= total} onClick={() => update({ page: String(page + 1) })}>
                Next <ChevronRight aria-hidden="true" />
              </Button>
            </div>
          </nav>
        )}
      </div>
    </>
  );
}
