import { useHealth } from "@/hooks/queries";
import { cn } from "@/lib/utils";

/** Small live status of the API: reachable, model loaded, calibrated. */
export function HealthIndicator({ compact = false }) {
  const { data, isPending, isError } = useHealth();
  let tone = "bg-muted-foreground";
  let label = "Checking server…";
  if (isError) {
    tone = "bg-destructive";
    label = "Server unreachable";
  } else if (data && !data.model_loaded) {
    tone = "bg-destructive";
    label = "Model not loaded";
  } else if (data && !data.calibrated) {
    tone = "bg-warning";
    label = "Uncalibrated: every result is referred";
  } else if (data) {
    tone = "bg-success";
    label = "Model ready";
  }
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground" role="status" title={label}>
      <span className="relative flex size-2.5 shrink-0">
        {!isPending && !isError && data?.calibrated && (
          <span className={cn("absolute inline-flex size-full animate-ping rounded-full opacity-40", tone)} />
        )}
        <span className={cn("relative inline-flex size-2.5 rounded-full", tone)} />
      </span>
      <span className={compact ? "sr-only" : undefined}>{label}</span>
    </div>
  );
}
