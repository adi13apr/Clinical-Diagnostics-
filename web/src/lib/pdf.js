/**
 * PDF report built in the browser. The patient name never leaves this device: the API doesn't
 * receive it (see the privacy page), so it only goes into this file.
 * jsPDF is loaded on first use to keep it out of the main bundle.
 */
import { CLASSES, formatDateTime, percent, probability, referralReason } from "@/lib/format";

const MARGIN = 48;
const INK = [15, 28, 46];
const MUTED = [85, 100, 118];
const PRIMARY = [14, 110, 140];
const WARNING_BG = [255, 244, 220];
const WARNING_BORDER = [242, 207, 134];
const WARNING_INK = [138, 83, 0];

// The built-in Helvetica only covers Latin-1; swap common typography for plain characters
function plain(text) {
  return String(text ?? "")
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[–—]/g, "-")
    .replace(/…/g, "...")
    .replace(/≥/g, ">=")
    .replace(/≤/g, "<=")
    .replace(/[^\x20-\x7E\xA0-\xFF\n]/g, "");
}

/**
 * Build and download the report PDF.
 * @param {object} args
 * @param {import("@/lib/api").Prediction} args.prediction
 * @param {import("@/lib/api").ReportResult|null} args.report AI report, if one was generated.
 * @param {{dataUrl: string, width: number, height: number}|null} args.image X-ray with the overlay.
 * @param {string} [args.patientName] Optional; stays on this device.
 * @param {string} [args.fileName]
 */
export async function downloadReportPdf({ prediction, report, image, patientName, fileName }) {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  const width = pageW - MARGIN * 2;
  let y = MARGIN;

  const ensure = (h) => {
    if (y + h > pageH - MARGIN) {
      doc.addPage();
      y = MARGIN;
    }
  };
  const text = (value, { size = 10, color = INK, bold = false, gap = 4, indent = 0 } = {}) => {
    doc.setFont("helvetica", bold ? "bold" : "normal");
    doc.setFontSize(size);
    doc.setTextColor(...color);
    const lines = doc.splitTextToSize(plain(value), width - indent);
    const lineH = size * 1.35;
    for (const line of lines) {
      ensure(lineH);
      doc.text(line, MARGIN + indent, y + size);
      y += lineH;
    }
    y += gap;
  };
  const heading = (value) => {
    ensure(36);
    y += 8;
    text(value, { size: 12, bold: true, color: PRIMARY, gap: 6 });
  };
  const row = (label, value) => {
    ensure(16);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(...MUTED);
    doc.text(plain(label), MARGIN, y + 10);
    doc.setTextColor(...INK);
    doc.setFont("helvetica", "bold");
    doc.text(plain(value), MARGIN + 170, y + 10);
    y += 16;
  };

  // Header
  doc.setFillColor(...PRIMARY);
  doc.rect(0, 0, pageW, 6, "F");
  text("Chest X-ray screening report", { size: 18, bold: true, gap: 2 });
  text(`Generated ${formatDateTime(new Date().toISOString())}`, { size: 9, color: MUTED, gap: 12 });

  heading("Details");
  if (patientName?.trim()) row("Patient", patientName.trim());
  row("Reference (trace ID)", prediction.trace_id);
  row("Model's most likely class", prediction.predicted_class);
  row("Calibrated confidence", probability(prediction.confidence));
  row("Triage", prediction.status === "refer" ? "Refer: clinician review recommended" : "Confident");
  y += 6;

  if (prediction.status === "refer") {
    const msg = referralReason(prediction);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    const lines = doc.splitTextToSize(plain(msg), width - 24);
    const boxH = 28 + lines.length * 13.5;
    ensure(boxH + 8);
    doc.setFillColor(...WARNING_BG);
    doc.setDrawColor(...WARNING_BORDER);
    doc.roundedRect(MARGIN, y, width, boxH, 6, 6, "FD");
    doc.setTextColor(...WARNING_INK);
    doc.setFont("helvetica", "bold");
    doc.text("Referral", MARGIN + 12, y + 17);
    doc.setFont("helvetica", "normal");
    lines.forEach((line, i) => doc.text(line, MARGIN + 12, y + 32 + i * 13.5));
    y += boxH + 10;
  }

  heading("Calibrated probabilities");
  for (const name of CLASSES) row(name, probability(prediction.probabilities[name]));
  text(
    `Results below ${percent(prediction.referral_threshold, 0)} confidence are referred for clinician review.`,
    { size: 9, color: MUTED, gap: 4 },
  );

  if (image) {
    heading("Where the model looked (Grad-CAM)");
    const maxW = width * 0.6;
    const maxH = 260;
    const scale = Math.min(maxW / image.width, maxH / image.height);
    const w = image.width * scale;
    const h = image.height * scale;
    ensure(h + 30);
    doc.addImage(image.dataUrl, "JPEG", MARGIN, y, w, h);
    y += h + 6;
    text("Red areas influenced the prediction most. This is a coarse map, not a lung segmentation.", {
      size: 9,
      color: MUTED,
    });
  }

  if (report) {
    const r = report.report;
    heading("AI report");
    text(r.summary, { gap: 8 });
    text("Recommendation", { bold: true, gap: 2 });
    text(r.recommendation, { gap: 8 });
    text("Limitations", { bold: true, gap: 2 });
    for (const item of r.limitations) text(`-  ${item}`, { indent: 6, gap: 2 });
    y += 6;
    text(report.source_line, { size: 9, color: MUTED });
  }

  heading("Disclaimer");
  text(
    report?.report.disclaimer ||
      "This result comes from an automated image classifier. It is not a diagnosis and must not be used " +
        "on its own to make clinical decisions. A qualified clinician must review the image and the " +
        "patient's clinical findings.",
    { size: 9, color: MUTED },
  );

  doc.save(fileName || `xray-report-${prediction.trace_id.slice(0, 8)}.pdf`);
}
