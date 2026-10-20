import { motion } from "framer-motion";
import { CLASS_COLORS, CLASSES, probability } from "@/lib/format";

/** One thin bar per class; the value is written at the end so colour isn't needed to read it. */
export function ProbabilityBars({ probabilities, predicted }) {
  return (
    <dl className="space-y-3">
      {CLASSES.map((name, i) => {
        const value = probabilities[name] ?? 0;
        return (
          <div key={name}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3 text-sm">
              <dt className={name === predicted ? "font-semibold" : "text-muted-foreground"}>{name}</dt>
              <dd className="tabular font-medium">{probability(value)}</dd>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted" aria-hidden="true">
              <motion.div
                className="h-full rounded-full"
                style={{ background: CLASS_COLORS[name] }}
                initial={{ width: 0 }}
                animate={{ width: `${Math.max(value * 100, 0.5)}%` }}
                transition={{ duration: 0.8, delay: 0.15 + i * 0.08, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
          </div>
        );
      })}
    </dl>
  );
}
