/**
 * Turn the flat span list from GET /traces/{id} into request groups for the waterfall.
 * Each root span (a request: prediction, Grad-CAM, report) gets its own time scale, because
 * the report may run minutes after the prediction.
 */

const EVENT_NAMES = new Set(["guard", "fallback_used", "report_cache_hit"]); // zero-duration markers

export function isEvent(span) {
  return EVENT_NAMES.has(span.name) || !span.latency_ms;
}

function startMs(span) {
  const t = span.start ? Date.parse(span.start) : NaN;
  return Number.isNaN(t) ? null : t;
}

/**
 * @param {import("@/lib/api").Span[]} spans
 * @returns {{root: object, rows: {span: object, depth: number, offset: number, width: number}[], startedAt: number|null, totalMs: number}[]}
 */
export function buildGroups(spans) {
  const byId = new Map(spans.map((s) => [s.span_id, s]));
  const children = new Map();
  const roots = [];
  for (const s of spans) {
    if (s.parent_span_id && byId.has(s.parent_span_id)) {
      if (!children.has(s.parent_span_id)) children.set(s.parent_span_id, []);
      children.get(s.parent_span_id).push(s);
    } else {
      roots.push(s);
    }
  }
  const byStart = (a, b) => (startMs(a) ?? 0) - (startMs(b) ?? 0);
  roots.sort(byStart);

  return roots.map((root) => {
    const flat = [];
    const walk = (span, depth) => {
      flat.push({ span, depth });
      (children.get(span.span_id) ?? []).sort(byStart).forEach((c) => walk(c, depth + 1));
    };
    walk(root, 0);

    const starts = flat.map((r) => startMs(r.span)).filter((t) => t !== null);
    const t0 = starts.length ? Math.min(...starts) : null;
    const end = Math.max(
      ...flat.map((r) => (startMs(r.span) ?? t0 ?? 0) + (r.span.latency_ms ?? 0)),
      (t0 ?? 0) + (root.latency_ms ?? 0),
    );
    const totalMs = Math.max(1, t0 === null ? root.latency_ms ?? 1 : end - t0);

    const rows = flat.map(({ span, depth }) => {
      const s = startMs(span);
      const offset = t0 === null || s === null ? 0 : (s - t0) / totalMs;
      const width = (span.latency_ms ?? 0) / totalMs;
      return { span, depth, offset: Math.min(1, Math.max(0, offset)), width: Math.min(1 - offset, Math.max(0, width)) };
    });
    return { root, rows, startedAt: t0, totalMs };
  });
}

/** Totals for the summary row. */
export function summarize(spans) {
  let input = 0;
  let output = 0;
  let cost = 0;
  let llmCalls = 0;
  for (const s of spans) {
    if (s.name === "llm_call") {
      const a = s.attributes ?? {};
      llmCalls += 1;
      input += a.input_tokens ?? 0;
      output += (a.output_tokens ?? 0) + (a.thinking_tokens ?? 0);
      cost += a.nominal_cost_usd ?? 0;
    }
  }
  return {
    spans: spans.length,
    errors: spans.filter((s) => s.status === "error").length,
    llmCalls,
    tokens: input + output,
    cost,
  };
}
