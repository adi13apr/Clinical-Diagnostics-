import { motion } from "framer-motion";
import { CheckCircle2, TriangleAlert } from "lucide-react";
import { percent, referralReason } from "@/lib/format";
import { cn } from "@/lib/utils";

/** Confident / Refer banner: icon, headline and the reason in words. */
export function TriageBanner({ prediction }) {
  const refer = prediction.status === "refer";
  const Icon = refer ? TriangleAlert : CheckCircle2;
  return (
    <motion.div
      role={refer ? "alert" : "status"}
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
      className={cn(
        "flex gap-3 rounded-xl border p-4",
        refer ? "border-warning-border bg-warning-soft text-warning" : "border-success/25 bg-success-soft text-success",
      )}
    >
      <Icon className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
      <div className="min-w-0">
        <p className="font-semibold">{refer ? "Refer: clinician review recommended" : "Confident result"}</p>
        <p className="mt-0.5 text-sm text-foreground/80">
          {refer
            ? referralReason(prediction)
            : `Confidence is at or above the ${percent(prediction.referral_threshold, 0)} referral threshold. A clinician should still review the image.`}
        </p>
        {!prediction.calibrated && (
          <p className="mt-2 text-sm font-medium text-foreground">
            The model&apos;s calibration file is missing on the server, so every result is referred.
          </p>
        )}
      </div>
    </motion.div>
  );
}
