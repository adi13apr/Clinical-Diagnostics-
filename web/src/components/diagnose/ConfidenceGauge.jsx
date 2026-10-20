import { animate, motion, useMotionValue, useReducedMotion, useTransform } from "framer-motion";
import { useEffect } from "react";
import { probability } from "@/lib/format";

const R = 80;
const STROKE = 14;
const CX = 100;
const CY = 96;
// Half circle from the left (0%) over the top to the right (100%)
const ARC = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`;

function pointAt(t, radius) {
  const angle = Math.PI * (1 - t);
  return [CX + radius * Math.cos(angle), CY - radius * Math.sin(angle)];
}

/**
 * Half-circle meter for the calibrated confidence. The fill colour carries the triage state
 * (confident / refer); the track is a lighter step of the same colour; a tick marks the referral
 * threshold. The number is also written out, so nothing depends on colour.
 */
export function ConfidenceGauge({ value, threshold = 0.9, status, size = "md", label = "Calibrated confidence" }) {
  const reduce = useReducedMotion();
  const progress = useMotionValue(reduce ? value : 0);
  const text = useTransform(progress, (v) => probability(v));

  useEffect(() => {
    const controls = animate(progress, value, { duration: reduce ? 0 : 1.1, ease: [0.16, 1, 0.3, 1] });
    return () => controls.stop();
  }, [value, progress, reduce]);

  const color = status === "refer" ? "var(--gauge-refer)" : "var(--gauge-confident)";
  const [t1x, t1y] = pointAt(threshold, R - STROKE / 2 - 3);
  const [t2x, t2y] = pointAt(threshold, R + STROKE / 2 + 3);
  const [lx, ly] = pointAt(threshold, R + STROKE / 2 + 16);
  const width = size === "sm" ? "w-40" : "w-56 sm:w-64";

  return (
    <figure className={`${width} mx-auto`} aria-label={`${label}: ${probability(value)}`} role="img">
      <svg viewBox="0 0 200 112" className="w-full overflow-visible" aria-hidden="true">
        <path
          d={ARC}
          fill="none"
          stroke={`color-mix(in oklab, ${color} 18%, transparent)`}
          strokeWidth={STROKE}
          strokeLinecap="round"
        />
        <motion.path
          d={ARC}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeLinecap="round"
          style={{ pathLength: progress }}
        />
        <line x1={t1x} y1={t1y} x2={t2x} y2={t2y} stroke="var(--foreground)" strokeWidth="2" strokeLinecap="round" />
        <text x={lx} y={ly} textAnchor="start" dominantBaseline="middle" className="fill-muted-foreground" fontSize="11">
          {Math.round(threshold * 100)}%
        </text>
        <motion.text
          x={CX}
          y={CY - 16}
          textAnchor="middle"
          className="fill-foreground font-semibold"
          fontSize={size === "sm" ? 26 : 28}
        >
          {text}
        </motion.text>
        <text x={CX} y={CY + 6} textAnchor="middle" className="fill-muted-foreground" fontSize="12">
          {label}
        </text>
      </svg>
    </figure>
  );
}
