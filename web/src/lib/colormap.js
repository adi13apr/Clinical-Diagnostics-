/**
 * Grad-CAM drawing that matches the Python overlay (services/heatmap_handler.py):
 * the 7x7 map is stretched over the whole image with bilinear resizing (cv2.INTER_LINEAR)
 * and coloured with the "jet" colormap (blue = low, red = high).
 */

/** Jet colormap: t in [0, 1] -> [r, g, b] 0-255. */
export function jet(t) {
  const x = Math.min(1, Math.max(0, t));
  const c = (v) => Math.round(255 * Math.min(1, Math.max(0, v)));
  return [c(1.5 - Math.abs(4 * x - 3)), c(1.5 - Math.abs(4 * x - 2)), c(1.5 - Math.abs(4 * x - 1))];
}

/** CSS gradient for the colour-scale legend. */
export const JET_GRADIENT = `linear-gradient(to right, ${[0, 0.125, 0.375, 0.625, 0.875, 1]
  .map((t) => `rgb(${jet(t).join(",")}) ${t * 100}%`)
  .join(", ")})`;

/** Bilinear sample with pixel-centre alignment, like cv2.resize(..., INTER_LINEAR). */
function sample(cam, rows, cols, y, x) {
  const sy = Math.min(rows - 1, Math.max(0, y));
  const sx = Math.min(cols - 1, Math.max(0, x));
  const y0 = Math.floor(sy);
  const x0 = Math.floor(sx);
  const y1 = Math.min(rows - 1, y0 + 1);
  const x1 = Math.min(cols - 1, x0 + 1);
  const fy = sy - y0;
  const fx = sx - x0;
  const top = cam[y0][x0] * (1 - fx) + cam[y0][x1] * fx;
  const bottom = cam[y1][x0] * (1 - fx) + cam[y1][x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

/**
 * Paint the map onto a canvas at the canvas's own resolution (CSS then stretches it over the image).
 * @param {HTMLCanvasElement} canvas
 * @param {number[][]} cam rows of values in [0, 1]
 */
export function drawCam(canvas, cam) {
  const rows = cam.length;
  const cols = cam[0]?.length ?? 0;
  if (!rows || !cols) return;
  const { width, height } = canvas;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(width, height);
  for (let py = 0; py < height; py++) {
    const y = (py + 0.5) * (rows / height) - 0.5;
    for (let px = 0; px < width; px++) {
      const x = (px + 0.5) * (cols / width) - 0.5;
      const [r, g, b] = jet(sample(cam, rows, cols, y, x));
      const i = (py * width + px) * 4;
      image.data[i] = r;
      image.data[i + 1] = g;
      image.data[i + 2] = b;
      image.data[i + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}

/**
 * The X-ray with the heatmap blended on top, as a JPEG data URL (for the PDF).
 * @param {HTMLImageElement} img a loaded image
 * @param {number[][]|null} cam
 * @param {number} opacity 0-1
 * @param {number} [maxSide] longest side of the output in pixels
 */
export function compositeDataUrl(img, cam, opacity, maxSide = 900) {
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  if (cam) {
    const heat = document.createElement("canvas");
    heat.width = 224; // the model's input size
    heat.height = 224;
    drawCam(heat, cam);
    ctx.globalAlpha = opacity;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(heat, 0, 0, w, h);
    ctx.globalAlpha = 1;
  }
  return { dataUrl: out.toDataURL("image/jpeg", 0.9), width: w, height: h };
}
