// Synthetic face renderer for front-end stability tests. No real faces: a
// skin-coloured ellipse with shading, a lateral light gradient, a fixed skin
// texture that moves with the head, dark eyes / brows / nostrils, lips and a
// non-skin background. Landmarks are placed so that the 10 shared MediaPipe
// patch polygons (README 4.2b) land on their canonical rectangles (4.2a); a
// landmark shared by several polygons sits at the mean of its placements.
import { PATCHES } from '../src/config';
import { gaussian, mulberry32 } from './synth';

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

const SKIN = [195, 145, 115], BG = [60, 72, 95], DARK = [45, 35, 32], BROW = [80, 58, 45], LIP = [160, 80, 78];
const PV = [0.33, 0.77, 0.53];

function hash2(i: number, j: number): number {
  let h = Math.imul(i, 374761393) ^ Math.imul(j, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296 - 0.5;
}

const smooth = (e: number) => (e <= 0 ? 0 : e >= 1 ? 1 : e * e * (3 - 2 * e));
/** Soft membership (1 inside, 0 outside, ~0.03 IOD blur) of an ellipse at (a, b) with radii (ra, rb). */
const blob = (u: number, v: number, a: number, b: number, ra: number, rb: number) =>
  1 - smooth((Math.sqrt(((u - a) / ra) ** 2 + ((v - b) / rb) ** 2) - 1) / (0.03 / Math.min(ra, rb)) + 0.5);

/**
 * Colour at canonical point (u, v) and its skin membership (0..1). Features
 * have soft edges and touch the patch borders as on a real face (brows at the
 * glabella / lateral forehead, nostrils under the nasal dorsum, lips under the
 * philtrum); a warm side light gives a coloured (not only luminance) gradient.
 */
function shade(u: number, v: number): [number, number, number, number] {
  const r2 = (u / 1.25) ** 2 + ((v - 0.3) / 1.65) ** 2;
  const face = 1 - smooth((Math.sqrt(r2) - 1) / 0.03 + 0.5);
  if (face <= 0) return [BG[0], BG[1], BG[2], 0];
  const au = Math.abs(u);
  const feats: [number, number[]][] = [
    [blob(au, v, 0.5, 0, 0.18, 0.08), DARK],                     // eyes
    [blob(au, v, 0.42, -0.27, 0.3, 0.045), BROW],                // brows (inner end at |u| ~ 0.12)
    [blob(au, v, 0.1, 0.4, 0.05, 0.035), DARK],                  // nostrils
    [blob(u, v, 0, 1.0, 0.35, 0.08), LIP],                       // lips
  ];
  let col = [...SKIN], skin = 1;
  for (const [m, c] of feats) if (m > 0) { for (let k = 0; k < 3; k++) col[k] = col[k] * (1 - m) + c[k] * m; skin *= 1 - m; }
  const sh = 1 - 0.35 * r2 * r2;
  const tex = 1 + 0.04 * hash2(Math.floor(u * 80), Math.floor(v * 80)); // ~1 px texture at IOD 70-80
  const out = [0, 1, 2].map((k) => (col[k] * sh * tex * (1 + 0.15 * u * WARM[k])) * face + BG[k] * (1 - face));
  return [out[0], out[1], out[2], skin * face];
}

const WARM = [1.3, 1.0, 0.6];

export interface Pose { cx: number; cy: number; iod: number; roll: number }

/**
 * Render the crop [ox, ox+cw) x [oy, oy+ch) of a W x H frame at time t.
 * pulse: relative pulse waveform value (multiplies the skin pulsatility vector).
 */
export function renderCrop(pose: Pose, ox: number, oy: number, cw: number, ch: number, pulse: number, noise: () => number, noiseSd: number): ImageData {
  const data = new Uint8ClampedArray(cw * ch * 4);
  const c = Math.cos(-pose.roll), s = Math.sin(-pose.roll);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const dx = (ox + x + 0.5 - pose.cx) / pose.iod, dy = (oy + y + 0.5 - pose.cy) / pose.iod;
    const u = c * dx - s * dy, v = s * dx + c * dy;
    const [r, g, b, skin] = shade(u, v);
    const p = 4 * (y * cw + x);
    const col = [r, g, b];
    for (let k = 0; k < 3; k++) data[p + k] = col[k] * (1 + skin * pulse * PV[k]) + noiseSd * noise();
    data[p + 3] = 255;
  }
  return { width: cw, height: ch, data, colorSpace: 'srgb' } as ImageData;
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
