import { cn } from "@/lib/utils";

/** Mark + wordmark. `compact` hides the wordmark (tablet rail). */
export function Logo({ compact = false, className }) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <svg viewBox="0 0 32 32" className="size-8 shrink-0" aria-hidden="true">
        <rect width="32" height="32" rx="8" className="fill-primary" />
        <path
          d="M16 7v18M10 11c2 1 4 1 6 0s4-1 6 0M9 16c2.4 1.2 4.7 1.2 7 0s4.6-1.2 7 0M10 21c2 1 4 1 6 0s4-1 6 0"
          fill="none"
          className="stroke-primary-foreground"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
      {!compact && (
        <span className="leading-tight">
          <span className="block text-sm font-semibold">Clinical Diagnostic</span>
          <span className="block text-xs text-muted-foreground">Chest X-ray portal</span>
        </span>
      )}
    </span>
  );
}
