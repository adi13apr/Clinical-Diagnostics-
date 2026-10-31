/**
 * Client for the FastAPI backend in api/ (see README "API").
 *
 * Every call except getHealth() sends the signed-in user's Supabase access token, and the server
 * queries Supabase as that user, so row-level security only ever returns their own rows.
 * Probabilities and confidences are 0-1 everywhere.
 */
import { supabase } from "@/lib/supabase";

export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

/**
 * @typedef {"Normal" | "Pneumonia" | "Tuberculosis"} ClassName
 * @typedef {"confident" | "refer"} Status
 */

/**
 * @typedef {object} Health
 * @property {string} status
 * @property {boolean} model_loaded
 * @property {boolean} calibrated
 * @property {number|null} temperature
 * @property {boolean} auth_configured
 * @property {boolean} llm_configured False: reports use the fixed template.
 * @property {"inline"|"queue"} report_mode "queue": reports are written by worker processes (REDIS_URL set).
 * @property {number|null} report_workers Workers listening for report jobs; null in inline mode.
 */

/**
 * @typedef {object} Prediction
 * @property {string} trace_id Use it for the Grad-CAM, report and trace calls.
 * @property {ClassName} predicted_class
 * @property {number} confidence Calibrated probability of the predicted class.
 * @property {Record<ClassName, number>} probabilities Calibrated probability per class.
 * @property {Status} status "refer" when confidence is below referral_threshold.
 * @property {number} referral_threshold
 * @property {string|null} referral_message
 * @property {number} temperature
 * @property {boolean} calibrated False: calibration file missing, so every result is "refer".
 * @property {number} inference_ms
 * @property {string[]} warnings Non-fatal problems, e.g. logging failed.
 */

/**
 * @typedef {object} GradcamMap
 * @property {string} trace_id
 * @property {ClassName} explained_class
 * @property {number[][]} cam 7x7 map, rows top to bottom, 0-1; stretch it over the whole image.
 * @property {{peak_region: string|null, region_shares: Record<string, number>, border_share: number|null,
 *   focus: "focal"|"diffuse"|"none", border_attention?: boolean, note?: string}} regions
 */

/**
 * @typedef {object} Report
 * @property {ClassName} predicted_class
 * @property {number} calibrated_probability
 * @property {Status} status
 * @property {string} summary
 * @property {string} recommendation
 * @property {string[]} limitations
 * @property {string} disclaimer
 */

/**
 * @typedef {object} ReportResult
 * @property {string} trace_id
 * @property {Report} report
 * @property {"llm"|"fallback"} source
 * @property {string|null} model
 * @property {string} source_line Who wrote the report, for display.
 * @property {string|null} reason Why the template was used (user-facing).
 * @property {boolean} cached
 * @property {"single"|"multi"} pipeline The report pipeline that actually ran (REPORT_PIPELINE).
 */

/**
 * @typedef {"queued" | "running" | "done" | "failed"} ReportJobStatus
 *
 * @typedef {object} ReportJob
 * @property {string|null} job_id Poll getReportJob() with it. Null when the server wrote the report during
 *   the request (no REDIS_URL): status is then "done" and result is already set.
 * @property {string} trace_id
 * @property {ReportJobStatus} status
 * @property {ReportResult|null} result The report, when status is "done".
 * @property {string|null} error Why the report failed (user-facing), when status is "failed".
 * @property {string[]} warnings
 */

/**
 * @typedef {object} HistoryItem
 * @property {number} id
 * @property {string} created_at ISO timestamp.
 * @property {ClassName} predicted_label
 * @property {number} confidence Calibrated if temperature is set, else raw.
 * @property {Status|null} status Null for rows logged before triage existed.
 * @property {number|null} temperature
 * @property {number} inference_ms
 * @property {string|null} trace_id
 *
 * @typedef {object} HistoryPage
 * @property {HistoryItem[]} items
 * @property {number} total
 * @property {number} limit
 * @property {number} offset
 */

/**
 * @typedef {object} DailyVolume
 * @property {string} day YYYY-MM-DD (UTC).
 * @property {number} predictions
 * @property {number} referred
 * @property {number} referral_rate
 * @property {Record<string, number>} by_label
 * @property {number|null} avg_inference_ms
 *
 * @typedef {object} Analytics
 * @property {number} days
 * @property {number} total_predictions
 * @property {number} total_referred
 * @property {number|null} referral_rate
 * @property {DailyVolume[]} daily Newest day first.
 * @property {object|null} pipeline Step latencies, tokens, guard and fallback rates; null if traces failed to load.
 * @property {string[]} warnings
 */

/**
 * @typedef {object} Span
 * @property {string} span_id
 * @property {string|null} parent_span_id
 * @property {string} name
 * @property {string|null} start ISO timestamp.
 * @property {number|null} latency_ms
 * @property {string} status "ok" or "error".
 * @property {string|null} error_type
 * @property {string|null} error
 * @property {Record<string, unknown>} attributes
 *
 * @typedef {object} Trace
 * @property {string} trace_id
 * @property {Span[]} spans Oldest first; parent_span_id gives the nesting.
 */

/** An HTTP error from the API (status 0 means the server couldn't be reached). */
export class ApiError extends Error {
  /**
   * @param {string} message User-facing message.
   * @param {number} status HTTP status, or 0 for a network failure.
   */
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function accessToken() {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession(); // refreshes an expired token first
  return data.session?.access_token ?? null;
}

function errorMessage(status, body) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return detail.map((d) => d.msg).join("; ");
  if (status === 401) return "Your session has expired. Please sign in again.";
  if (status >= 500) return "The server had a problem. Please try again in a moment.";
  return `Request failed (${status})`;
}

/**
 * Low-level request helper: adds the bearer token and turns failures into ApiError.
 * @param {string} path Path on the API, e.g. "/history?limit=20".
 * @param {{method?: string, body?: BodyInit, auth?: boolean, signal?: AbortSignal, as?: "json"|"blob"}} [options]
 * @returns {Promise<any>}
 */
async function request(path, { method = "GET", body, auth = true, signal, as = "json" } = {}) {
  const headers = {};
  if (auth) {
    const token = await accessToken();
    if (!token) throw new ApiError("Please sign in again.", 401);
    headers.Authorization = `Bearer ${token}`;
  }
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, { method, body, headers, signal });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError(`Can't reach the diagnosis server at ${API_URL}. Is it running?`, 0);
  }
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new ApiError(errorMessage(res.status, data), res.status);
  }
  return as === "blob" ? res.blob() : res.json();
}

function query(params) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

/**
 * Is the model loaded and calibrated, is the LLM configured. No login needed.
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<Health>}
 */
export function getHealth({ signal } = {}) {
  return request("/health", { auth: false, signal });
}

/**
 * Classify one chest X-ray (JPEG or PNG, max 10 MB by default).
 * The server keeps the image in memory for an hour so Grad-CAM and the report can use it.
 * @param {File} file
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<Prediction>}
 */
export function predict(file, { signal } = {}) {
  const form = new FormData();
  form.append("file", file);
  return request("/predict", { method: "POST", body: form, signal });
}

/**
 * The raw 7x7 Grad-CAM map for the predicted class plus a region summary, for drawing an overlay.
 * @param {string} traceId From predict().
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<GradcamMap>}
 */
export function getGradcamMap(traceId, { signal } = {}) {
  return request(`/gradcam/${encodeURIComponent(traceId)}/map`, { signal });
}

/**
 * The server-rendered 3-panel Grad-CAM figure (original, overlay, heatmap) as a PNG blob.
 * Fetched with the token, so it can't be an <img src> URL directly; use URL.createObjectURL().
 * @param {string} traceId From predict().
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<Blob>}
 */
export function getGradcamFigure(traceId, { signal } = {}) {
  return request(`/gradcam/${encodeURIComponent(traceId)}`, { signal, as: "blob" });
}

/**
 * Start a plain-language AI report for a prediction. The server checks it against the classifier and falls
 * back to a fixed template if needed.
 *
 * With a report worker (the server has REDIS_URL) this returns at once with a queued job: poll
 * getReportJob() until it's "done" or "failed". Asking again while that job is queued or running returns
 * the same job. Without a worker the server writes the report during the request (up to a minute on the
 * LLM's free tier) and returns it as a finished job with no job_id.
 * @param {string} traceId From predict().
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<ReportJob>}
 */
export function createReport(traceId, { signal } = {}) {
  return request(`/report/${encodeURIComponent(traceId)}`, { method: "POST", signal });
}

/**
 * Status of a report job, with the report once it's done. Only the user who started it can read it (404
 * otherwise, and for expired jobs).
 * @param {string} jobId From createReport().
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<ReportJob>}
 */
export function getReportJob(jobId, { signal } = {}) {
  return request(`/report/jobs/${encodeURIComponent(jobId)}`, { signal });
}

/**
 * One page of the user's past predictions, newest first. Dates are UTC days, inclusive.
 * @param {{limit?: number, offset?: number, label?: ClassName, status?: Status,
 *   dateFrom?: string, dateTo?: string}} [filters] Dates as YYYY-MM-DD.
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<HistoryPage>}
 */
export function getHistory({ limit = 20, offset = 0, label, status, dateFrom, dateTo } = {}, { signal } = {}) {
  return request(
    `/history${query({ limit, offset, label, status, date_from: dateFrom, date_to: dateTo })}`,
    { signal },
  );
}

/**
 * Daily volume, referral rate and pipeline metrics for the last `days` days.
 * @param {{days?: number}} [params] 1-365, default 30.
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<Analytics>}
 */
export function getAnalytics({ days = 30 } = {}, { signal } = {}) {
  return request(`/analytics${query({ days })}`, { signal });
}

/**
 * Every recorded step of one prediction (and its Grad-CAM and report), for the trace viewer.
 * @param {string} traceId
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<Trace>}
 */
export function getTrace(traceId, { signal } = {}) {
  return request(`/traces/${encodeURIComponent(traceId)}`, { signal });
}
