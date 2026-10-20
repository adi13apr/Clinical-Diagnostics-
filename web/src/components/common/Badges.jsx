import { CheckCircle2, CircleHelp, TriangleAlert } from "lucide-react";
import { CLASS_COLORS } from "@/lib/format";
import { cn } from "@/lib/utils";

/** Triage status with an icon and a word, so it never relies on colour alone. */
export function StatusBadge({ status, className }) {
  if (status === "confident") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full bg-success-soft px-2.5 py-0.5 text-xs font-medium text-success",
          className,
        )}
      >
        <CheckCircle2 className="size-3.5" aria-hidden="true" /> Confident
      </span>
    );
  }
  if (status === "refer") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full bg-warning-soft px-2.5 py-0.5 text-xs font-medium text-warning",
          className,
        )}
      >
        <TriangleAlert className="size-3.5" aria-hidden="true" /> Refer
      </span>
    );
  }
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs text-muted-foreground", className)}>
      <CircleHelp className="size-3.5" aria-hidden="true" /> Not recorded
    </span>
  );
}

/** Class name with its colour dot (the dot carries identity; the text stays in ink colour). */
export function ClassLabel({ name, className }) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <span
        className="size-2.5 shrink-0 rounded-full"
        style={{ background: CLASS_COLORS[name] ?? "var(--muted-foreground)" }}
        aria-hidden="true"
      />
      {name}
    </span>
  );
}
