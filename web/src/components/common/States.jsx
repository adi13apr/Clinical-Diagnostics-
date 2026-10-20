import { AlertCircle, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** Nothing to show yet: an icon, a sentence and an optional next step. */
export function EmptyState({ icon: Icon, title, description, action, className }) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border border-dashed px-6 py-12 text-center",
        className,
      )}
    >
      {Icon && (
        <div className="mb-4 flex size-12 items-center justify-center rounded-full bg-accent text-accent-foreground">
          <Icon className="size-6" aria-hidden="true" />
        </div>
      )}
      <h2 className="text-base font-semibold">{title}</h2>
      {description && <p className="mt-1 max-w-sm text-sm text-muted-foreground">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

/** A failed request, in words, with a retry button. */
export function ErrorState({ title = "Something went wrong", error, onRetry, className }) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border border-destructive/30 bg-destructive/5 px-6 py-10 text-center",
        className,
      )}
    >
      <AlertCircle className="mb-3 size-8 text-destructive" aria-hidden="true" />
      <h2 className="text-base font-semibold">{title}</h2>
      {error && <p className="mt-1 max-w-md text-sm text-muted-foreground">{error.message || String(error)}</p>}
      {onRetry && (
        <Button variant="outline" size="sm" className="mt-4" onClick={onRetry}>
          <RotateCw aria-hidden="true" /> Try again
        </Button>
      )}
    </div>
  );
}
