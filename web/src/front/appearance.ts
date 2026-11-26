// appearance.ts — target appearance gate (README 4.1, 11.2.4): a colour
// histogram of the target's central face area; a re-acquisition candidate
// whose Bhattacharyya distance to the target model exceeds the threshold is
// rejected, so a bystander crossing the target position is not adopted.
//
// Histogram: 8 x 8 bins of normalised chromaticity (r, g) = (R, G) / (R+G+B)
// over [0.2, 0.6) x [0.2, 0.45), plus 4 soft brightness bins -> 256 bins. Chroma
// carries skin tone / clothing-independent identity; the coarse brightness
// axis separates e.g. hair or a dark background without being thrown by small
// exposure changes.
import type { Box } from './geometry';

export const HIST_BINS = 8 * 8 * 4;

/** Colour histogram (sums to 1) of the central `frac` of box in an RGBA image (pixel coords). */
export function centralHistogram(
  data: Uint8ClampedArray, w: number, h: number, box: Box, frac = 0.5, step = 1,
): Float64Array {
  const hist = new Float64Array(HIST_BINS);
  const x0 = Math.max(0, Math.floor(box.x + (box.w * (1 - frac)) / 2)), x1 = Math.min(w, Math.ceil(box.x + (box.w * (1 + frac)) / 2));
  const y0 = Math.max(0, Math.floor(box.y + (box.h * (1 - frac)) / 2)), y1 = Math.min(h, Math.ceil(box.y + (box.h * (1 + frac)) / 2));
  let n = 0;
  for (let y = y0; y < y1; y += step) for (let x = x0; x < x1; x += step) {
    const p = 4 * (y * w + x);
    const R = data[p], G = data[p + 1], B = data[p + 2], S = R + G + B;
    if (S < 30) continue; // too dark for a chromaticity
    const r = R / S, g = G / S;
    const ir = Math.min(7, Math.max(0, Math.floor((r - 0.2) / 0.05)));
    const ig = Math.min(7, Math.max(0, Math.floor((g - 0.2) / 0.03125)));
    // brightness: soft (linear) assignment between bin centres so a small
    // exposure change moves mass gradually instead of across a hard edge
    const bf = Math.min(3, Math.max(0, S / 192 - 0.5));
    const b0 = Math.floor(bf), wb = bf - b0;
    hist[(b0 * 8 + ir) * 8 + ig] += 1 - wb;
    if (wb > 0) hist[(Math.min(3, b0 + 1) * 8 + ir) * 8 + ig] += wb;
    n++;
  }
  if (n) for (let i = 0; i < HIST_BINS; i++) hist[i] /= n;
  return hist;
}

/** Bhattacharyya distance sqrt(1 - sum sqrt(p q)) in [0, 1] (OpenCV HISTCMP_BHATTACHARYYA). */
export function bhattacharyya(p: Float64Array, q: Float64Array): number {
  let bc = 0, sp = 0, sq = 0;
  for (let i = 0; i < p.length; i++) { bc += Math.sqrt(p[i] * q[i]); sp += p[i]; sq += q[i]; }
  if (!(sp > 0 && sq > 0)) return 1;
  return Math.sqrt(Math.max(0, 1 - bc / Math.sqrt(sp * sq)));
}

/** Target appearance model with a slow EMA update while the target is tracked. */
export class AppearanceGate {
  private model: Float64Array | null = null;

  constructor(private maxDistance: number, private emaAlpha = 0.05) {}

  get hasModel() { return this.model !== null; }

  reset() { this.model = null; }

  /** Distance of a candidate histogram to the model (0 if no model yet). */
  distance(h: Float64Array): number {
    return this.model ? bhattacharyya(this.model, h) : 0;
  }

  /** True if the candidate may be adopted as the target. */
  accept(h: Float64Array): boolean {
    return !this.model || this.distance(h) <= this.maxDistance;
  }

  /** Learn from a confirmed target observation. */
  update(h: Float64Array) {
    if (!this.model) { this.model = Float64Array.from(h); return; }
    const a = this.emaAlpha;
    for (let i = 0; i < h.length; i++) this.model[i] = (1 - a) * this.model[i] + a * h[i];
  }
}
