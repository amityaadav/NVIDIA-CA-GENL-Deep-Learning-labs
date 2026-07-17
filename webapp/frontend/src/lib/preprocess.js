/**
 * Turn a raw trackpad drawing into the 784-value input MNIST expects.
 *
 * This is the step that decides whether predictions are any good. MNIST digits
 * are white-on-black, cropped to the ink, scaled to ~20px, and centered by
 * center-of-mass inside a 28x28 field. A user draws black-on-white, off-center,
 * at arbitrary size -- so we replicate the MNIST normalization here:
 *
 *   1. read the drawing, convert to an ink map (0 = blank, 1 = full ink)
 *   2. find the ink bounding box and center of mass
 *   3. scale the longest side to 20px, preserving aspect
 *   4. paste it into a 28x28 field so the center of mass lands at the middle
 *   5. read back 28x28 grayscale as white-on-black -> 784 floats in [0, 1]
 *
 * `normalize` returns the 784 input plus the intermediate geometry and a
 * white-on-black ink canvas, so the UI can animate the crop/scale/center.
 * `canvasToInput` is the thin wrapper that returns just the input (or null).
 */
export function normalize(sourceCanvas) {
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  const ctx = sourceCanvas.getContext("2d");
  const { data } = ctx.getImageData(0, 0, w, h);

  // Ink map: the drawing is dark strokes on a white background, so ink is the
  // inverse of luminance, weighted by alpha in case the background is clear.
  const ink = new Float32Array(w * h);
  let minX = w, minY = h, maxX = -1, maxY = -1;
  const INK_THRESHOLD = 0.08;

  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2], a = data[i * 4 + 3] / 255;
    const luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    const v = a * (1 - luma); // dark, opaque pixel -> ~1
    ink[i] = v;
    if (v > INK_THRESHOLD) {
      const x = i % w, y = (i / w) | 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  if (maxX < 0) return null; // nothing drawn

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;

  // Center of mass within the bounding box (weighted by ink).
  let sum = 0, comX = 0, comY = 0;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const v = ink[y * w + x];
      sum += v;
      comX += v * x;
      comY += v * y;
    }
  }
  comX /= sum;
  comY /= sum;

  // Scale so the longer side of the ink becomes 20px inside the 28px field.
  const scale = 20 / Math.max(boxW, boxH);

  // Build a white-on-black source so we can drawImage with proper interpolation.
  const src = document.createElement("canvas");
  src.width = w;
  src.height = h;
  const sctx = src.getContext("2d");
  const out = sctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.min(255, Math.round(ink[i] * 255));
    out.data[i * 4] = v;
    out.data[i * 4 + 1] = v;
    out.data[i * 4 + 2] = v;
    out.data[i * 4 + 3] = 255;
  }
  sctx.putImageData(out, 0, 0);

  // Paste the scaled crop so the center of mass lands at (14, 14).
  const dst = document.createElement("canvas");
  dst.width = 28;
  dst.height = 28;
  const dctx = dst.getContext("2d");
  dctx.fillStyle = "black";
  dctx.fillRect(0, 0, 28, 28);
  dctx.imageSmoothingEnabled = true;
  dctx.imageSmoothingQuality = "high";

  const destX = 14 - (comX - minX) * scale;
  const destY = 14 - (comY - minY) * scale;
  dctx.drawImage(src, minX, minY, boxW, boxH, destX, destY, boxW * scale, boxH * scale);

  const px = dctx.getImageData(0, 0, 28, 28).data;
  const input = new Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    input[i] = px[i * 4] / 255; // white-on-black, so brightness == ink
  }

  return {
    input,
    // Everything the normalization animation needs to replay crop -> scale -> center.
    geometry: { w, h, minX, minY, boxW, boxH, comX, comY, scale },
    inkCanvas: src, // white-on-black, full source size
  };
}

/** Just the 784 input (or null if the canvas is effectively blank). */
export function canvasToInput(sourceCanvas) {
  const result = normalize(sourceCanvas);
  return result ? result.input : null;
}
