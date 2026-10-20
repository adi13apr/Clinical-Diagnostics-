/** Class order is fixed by the trained weights (services/model_handler.py). */
export const CLASSES = ["Normal", "Pneumonia", "Tuberculosis"];

/** CSS variables from index.css, validated for colour-blind separation in light and dark. */
export const CLASS_COLORS = {
  Normal: "var(--chart-normal)",
  Pneumonia: "var(--chart-pneumonia)",
  Tuberculosis: "var(--chart-tuberculosis)",
};

/** Plain-language description shown next to each class. */
export const CLASS_BLURBS = {
  Normal: "No pattern of pneumonia or tuberculosis found by the model.",
  Pneumonia: "A pattern the model associates with pneumonia.",
  Tuberculosis: "A pattern the model associates with tuberculosis.",
};

/** Friendly names for trace span names (services/tracing.py). */
export const STEP_LABELS = {
  prediction: "Prediction (total)",
  preprocess: "Preprocess",
  classify: "Classify",
  calibrate: "Calibrate",
  log_prediction: "Log prediction",
  save_history: "Save history",
  gradcam: "Grad-CAM",
  report: "Report (total)",
  llm_call: "LLM call",
  tool_call: "Tool call",
  guard: "Guard check",
  fallback_used: "Template fallback",
  report_cache_hit: "Cache hit",
};

/**
 * 0.9734 -> "97.3%".
 * @param {number|null|undefined} value 0-1
 * @param {number} [digits]
 */
export function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * A model probability, never shown as a certainty it doesn't have:
 * 0.99996 -> "99.99%" (not "100.0%"), 0.00003 -> "<0.1%" (not "0.0%"), 0.9734 -> "97.3%".
 * Rounds like the report agent's run_classifier (4 dp), so the page and the report text agree.
 * @param {number|null|undefined} value 0-1
 */
export function probability(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value >= 1) return "100%";
  if (value <= 0) return "0%";
  if (value < 0.0005) return "<0.1%";
  if (value >= 0.9995) return `${Math.min(99.99, Math.round(value * 10000) / 100).toFixed(2)}%`;
  return `${Math.min(99.9, Math.round(value * 1000) / 10).toFixed(1)}%`;
}

/** Why a result was referred, in one sentence (the API's referral_message is a terse status string). */
export function referralReason(prediction) {
  if (prediction.confidence >= prediction.referral_threshold) {
    // Only happens when the server's calibration file is missing: then everything is referred
    return "Every result is referred while the model's calibration is missing. A clinician should review this image.";
  }
  return `Confidence ${probability(prediction.confidence)} is below the ${percent(prediction.referral_threshold, 0)} threshold, so the model isn't sure enough. A clinician should review this image before any decision.`;
}

/** 1234.5 -> "1.23 s", 87 -> "87 ms". */
export function duration(ms) {
  if (ms === null || ms === undefined) return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s`;
  if (ms > 0 && ms < 1) return "<1 ms";
  return `${Math.round(ms)} ms`;
}

/** 1284 -> "1,284"; 12900 -> "12.9K". */
export function compact(n) {
  if (n === null || n === undefined) return "—";
  return new Intl.NumberFormat(undefined, {
    notation: n >= 10000 ? "compact" : "standard",
    maximumFractionDigits: 1,
  }).format(n);
}

const dateTime = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const dayFormat = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", timeZone: "UTC" });

/** ISO timestamp -> local "20 Oct 2026, 14:05". */
export function formatDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : dateTime.format(d);
}

/** "2026-10-20" (a UTC day) -> "Oct 20". */
export function formatDay(day) {
  return dayFormat.format(new Date(`${day}T00:00:00Z`));
}

/** First 8 characters of an id for compact display; keep the full id in title/aria. */
export function shortId(id) {
  return id ? id.slice(0, 8) : "—";
}
