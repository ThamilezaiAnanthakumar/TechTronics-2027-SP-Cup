// geometry.ts — patch polygons from MediaPipe landmarks (README 4.2b),
// polygon rasterisation, face box and eye-line pose.
import { PATCHES } from '../config';

/** Landmarks as interleaved normalised x,y (length 2 * nLandmarks). */
export type Landmarks = Float32Array;

export interface Box { x: number; y: number; w: number; h: number }

export function landmarksBox(lm: Landmarks): Box {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < lm.length; i += 2) {
    x0 = Math.min(x0, lm[i]); x1 = Math.max(x1, lm[i]);
    y0 = Math.min(y0, lm[i + 1]); y1 = Math.max(y1, lm[i + 1]);
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Patch polygons in pixel coordinates (vertex order exactly as published). */
export function patchPolygons(lm: Landmarks, W: number, H: number): Float32Array[] {
  return PATCHES.map((p) => {
    const poly = new Float32Array(p.mediapipe.length * 2);
    p.mediapipe.forEach((k, i) => {
      poly[2 * i] = lm[2 * k] * W;
      poly[2 * i + 1] = lm[2 * k + 1] * H;
    });
    return poly;
  });
}

/** Eye centres (mean of the eye corners 33/133 and 362/263), IOD and roll. */
export function eyePose(lm: Landmarks, W: number, H: number) {
  const pt = (k: number) => [lm[2 * k] * W, lm[2 * k + 1] * H];
  const [a, b, c, d] = [pt(33), pt(133), pt(362), pt(263)];
  const e1 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], e2 = [(c[0] + d[0]) / 2, (c[1] + d[1]) / 2];
  const iod = Math.hypot(e2[0] - e1[0], e2[1] - e1[1]);
  return { cx: (e1[0] + e2[0]) / 2, cy: (e1[1] + e2[1]) / 2, iod, roll: Math.atan2(e2[1] - e1[1], e2[0] - e1[0]) };
}

/**
 * Scanline rasterisation with the non-zero winding rule (published vertex
 * order may self-intersect slightly; non-zero avoids holes). Pixel (x, y) is
 * inside when its centre (x + 0.5, y + 0.5) is. Returns indices into a crop of
 * width cw / height ch whose origin is (ox, oy) in pixel coordinates.
 */
export function rasterise(poly: Float32Array, ox: number, oy: number, cw: number, ch: number): Int32Array {
  const n = poly.length / 2;
  let ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) { ymin = Math.min(ymin, poly[2 * i + 1]); ymax = Math.max(ymax, poly[2 * i + 1]); }
  const r0 = Math.max(0, Math.floor(ymin - oy)), r1 = Math.min(ch - 1, Math.ceil(ymax - oy));
  const out: number[] = [];
  const xs: number[] = [], ws: number[] = [];
  for (let r = r0; r <= r1; r++) {
    const yc = oy + r + 0.5;
    xs.length = 0; ws.length = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const y1 = poly[2 * i + 1], y2 = poly[2 * j + 1];
      if ((y1 <= yc && y2 > yc) || (y2 <= yc && y1 > yc)) {
        const x1 = poly[2 * i], x2 = poly[2 * j];
        xs.push(x1 + ((yc - y1) / (y2 - y1)) * (x2 - x1));
        ws.push(y2 > y1 ? 1 : -1);
      }
    }
    if (xs.length < 2) continue;
    const order = xs.map((_, k) => k).sort((p, q) => xs[p] - xs[q]);
    let wind = 0;
    for (let k = 0; k < order.length - 1; k++) {
      wind += ws[order[k]];
      if (wind === 0) continue;
      const xa = xs[order[k]], xb = xs[order[k + 1]];
      const c0 = Math.max(0, Math.ceil(xa - ox - 0.5)), c1 = Math.min(cw - 1, Math.floor(xb - ox - 0.5));
      for (let c = c0; c <= c1; c++) out.push(r * cw + c);
    }
  }
  return Int32Array.from(out);
}
