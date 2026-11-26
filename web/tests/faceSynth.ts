// Synthetic face renderer for front-end stability tests. No real faces: a
// skin-coloured ellipse with shading, a lateral light gradient, a fixed skin
// texture that moves with the head, dark eyes / brows / nostrils, lips and a
// non-skin background. Landmarks are placed so that the 10 shared MediaPipe
// patch polygons (README 4.2b) land on their canonical rectangles (4.2a); a
// landmark shared by several polygons sits at the mean of its placements.
import { PATCHES } from '../src/config';
import { gaussian, mulberry32 } from './synth';
import { renderCrop, type Pose } from './faceRender';

export { renderCrop, type Pose };

export const N_LM = 478;

/** Canonical landmark positions (IOD units, origin = eye midpoint, +y down). */
export function canonicalLandmarks(): Float64Array {
  const sum = new Float64Array(N_LM * 2), cnt = new Float64Array(N_LM);
  const put = (k: number, u: number, v: number) => { sum[2 * k] += u; sum[2 * k + 1] += v; cnt[k]++; };
  for (const p of PATCHES) {
    const { cx, cy, w, h } = p.canonical;
    const n = p.mediapipe.length;
    p.mediapipe.forEach((k, i) => {
      const a = (2 * Math.PI * i) / n;
      // superellipse (|cos|^0.5) so the polygon fills most of the rectangle
      const c = Math.cos(a), s = Math.sin(a);
      put(k, cx + (w / 2) * Math.sign(c) * Math.sqrt(Math.abs(c)), cy + (h / 2) * Math.sign(s) * Math.sqrt(Math.abs(s)));
    });
  }
  // eye corners used for the pose (33/133 one eye, 362/263 the other)
  put(33, -0.65, 0); put(133, -0.35, 0); put(362, 0.35, 0); put(263, 0.65, 0);
  const out = new Float64Array(N_LM * 2);
  for (let k = 0; k < N_LM; k++) {
    if (cnt[k]) { out[2 * k] = sum[2 * k] / cnt[k]; out[2 * k + 1] = sum[2 * k + 1] / cnt[k]; }
    else { out[2 * k] = 0; out[2 * k + 1] = 0.3; } // unused landmarks: face centre
  }
  return out;
}

/** Landmarks (normalised to W x H) for a pose, with per-landmark and common jitter (px). */
export function poseLandmarks(can: Float64Array, pose: Pose, W: number, H: number, jit: { common: number; indep: number; ar?: number }, g: () => number, state?: Float64Array): Float32Array {
  const out = new Float32Array(N_LM * 2);
  const c = Math.cos(pose.roll), s = Math.sin(pose.roll);
  // jitter: white, or AR(1) with the same marginal SD when ar is set
  const a = jit.ar ?? 0, k1 = Math.sqrt(1 - a * a);
  const st = state ?? new Float64Array(N_LM * 2 + 2);
  const nx = (i: number, sd: number) => (st[i] = a * st[i] + k1 * sd * g());
  const jx = nx(N_LM * 2, jit.common), jy = nx(N_LM * 2 + 1, jit.common);
  for (let k = 0; k < N_LM; k++) {
    const u = can[2 * k], v = can[2 * k + 1];
    const x = pose.cx + pose.iod * (c * u - s * v) + jx + nx(2 * k, jit.indep);
    const y = pose.cy + pose.iod * (s * u + c * v) + jy + nx(2 * k + 1, jit.indep);
    out[2 * k] = x / W; out[2 * k + 1] = y / H;
  }
  return out;
}

export interface FaceSeqOpts {
  seconds?: number;
  fps?: number;
  bpm?: number;
  pulseAmp?: number;   // relative amplitude on G
  noiseSd?: number;    // per-pixel sensor noise, 0..255 units
  iod?: number;        // px
  jitter?: { common: number; indep: number; ar?: number }; // px (Gaussian SD); ar: AR(1) coefficient per frame (temporal correlation)
  drift?: { px: number; hz: number; rollRad: number }; // slow head drift (image and landmarks)
  seed?: number;
}

/** Frame generator: yields the ImageData crop, its origin and landmarks. */
export function* faceSequence(o: FaceSeqOpts = {}) {
  const { seconds = 40, fps = 15, bpm = 72, pulseAmp = 0.004, noiseSd = 3, iod = 70,
    jitter = { common: 0, indep: 0 }, drift = { px: 0, hz: 0.05, rollRad: 0 }, seed = 11 } = o;
  const W = 640, H = 480;
  const can = canonicalLandmarks();
  const g = gaussian(mulberry32(seed)), gn = gaussian(mulberry32(seed + 1));
  const n = Math.round(seconds * fps);
  const jstate = new Float64Array(N_LM * 2 + 2);
  for (let i = 0; i < n; i++) {
    const t = i / fps;
    const ph = 2 * Math.PI * drift.hz * t;
    const pose: Pose = { cx: 320 + drift.px * Math.sin(ph), cy: 220 + 0.5 * drift.px * Math.sin(0.7 * ph), iod, roll: drift.rollRad * Math.sin(0.5 * ph) };
    const lm = poseLandmarks(can, pose, W, H, jitter, g, jstate);
    // crop as the worker does: union of the patch polygons + 3 px
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of PATCHES) for (const k of p.mediapipe) {
      x0 = Math.min(x0, lm[2 * k] * W); x1 = Math.max(x1, lm[2 * k] * W);
      y0 = Math.min(y0, lm[2 * k + 1] * H); y1 = Math.max(y1, lm[2 * k + 1] * H);
    }
    const ox = Math.max(0, Math.floor(x0) - 3), oy = Math.max(0, Math.floor(y0) - 3);
    const cw = Math.min(W, Math.ceil(x1) + 3) - ox, ch = Math.min(H, Math.ceil(y1) + 3) - oy;
    const s = Math.sin(2 * Math.PI * (bpm / 60) * t) + 0.3 * Math.sin(2 * Math.PI * (2 * bpm / 60) * t + 1);
    const img = renderCrop(pose, ox, oy, cw, ch, (pulseAmp / 0.77) * s, gn, noiseSd);
    yield { t, img, ox, oy, W, H, lm };
  }
}
