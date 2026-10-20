import { ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";

/** The standing reminder that this is a screening aid. */
export function Disclaimer({ className }) {
  return (
    <p className={cn("flex items-start gap-2 text-xs leading-relaxed text-muted-foreground", className)}>
      <ShieldAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
      <span>
        Research prototype, not a medical device. Results are a screening aid, not a diagnosis; a qualified
        clinician must review every image.
      </span>
    </p>
  );
}
