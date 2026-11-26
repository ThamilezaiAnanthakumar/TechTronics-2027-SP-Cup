// draw.ts — canvas drawing for the overlay, BVP waveform and spectrogram.
import { PATCHES } from '../config';
import type { CoreTick, HrRow } from '../core/types';
import type { FrameInfo } from '../front/frontEnd';
import { patchPolygons, type Box, type Landmarks } from '../front/geometry';

export const STATUS_COLOR: Record<string, string> = {
  OK: '#2fbf71', MASKED: '#3aa7d9', WARMUP: '#8a94a6', LOW_SNR: '#e0a526', LOW_COVERAGE: '#e0a526', NO_FACE: '#e5484d',
};

function fitCanvas(c: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const g = c.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w: c.clientWidth, h: c.clientHeight };
}

/** Weight 0..1 -> colour (cool -> warm). */
function weightColor(w: number, alpha: number) {
  const x = Math.max(0, Math.min(1, w / 0.35));
  const hue = 210 - 200 * x; // blue -> red
  return `hsla(${hue}, 85%, 55%, ${alpha})`;
}

export function drawOverlay(
  c: HTMLCanvasElement, W: number, H: number, faces: Box[], target: number,
  lm: Landmarks | null, info: FrameInfo | null, tick: CoreTick | null,
) {
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const g = c.getContext('2d')!;
  g.clearRect(0, 0, W, H);
  const lw = Math.max(1.5, W / 320);
  faces.forEach((b, i) => {
    if (i === target) return;
    g.setLineDash([6 * lw, 4 * lw]);
    g.strokeStyle = 'rgba(255,255,255,0.75)';
    g.lineWidth = lw;
    g.strokeRect(b.x * W, b.y * H, b.w * W, b.h * H);
  });
  g.setLineDash([]);
  if (target < 0 || !lm) return;
  const b = faces[target];
  g.strokeStyle = '#2fbf71';
  g.lineWidth = 2 * lw;
  g.strokeRect(b.x * W, b.y * H, b.w * W, b.h * H);

  const polys = patchPolygons(lm, W, H);
  const weights = new Map(tick?.patches.map((p) => [p.id, p]) ?? []);
  PATCHES.forEach((def, i) => {
    const poly = polys[i];
    g.beginPath();
    for (let k = 0; k < poly.length; k += 2) (k ? g.lineTo : g.moveTo).call(g, poly[k], poly[k + 1]);
    g.closePath();
    const occluded = info?.occluded[i] ?? false;
    const valid = info?.valid[i] ?? false;
    const q = weights.get(def.id);
    if (occluded || !valid) {
      g.fillStyle = 'rgba(110,110,110,0.55)';
      g.fill('nonzero');
      g.strokeStyle = 'rgba(160,160,160,0.9)';
    } else if (q?.used) {
      g.fillStyle = weightColor(q.weight, 0.45);
      g.fill('nonzero');
      g.strokeStyle = weightColor(q.weight, 1);
    } else {
      g.strokeStyle = 'rgba(255,255,255,0.55)';
    }
    g.lineWidth = lw;
    g.stroke();
  });
}

export function drawBvp(c: HTMLCanvasElement, bvp: Float64Array | null, color: string) {
  const { g, w, h } = fitCanvas(c);
  g.clearRect(0, 0, w, h);
  g.strokeStyle = 'rgba(128,128,128,0.25)';
  g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
  if (!bvp || bvp.length < 2) return;
  let m = 0;
  for (const v of bvp) m = Math.max(m, Math.abs(v));
  m = m || 1;
  g.strokeStyle = color;
  g.lineWidth = 2;
  g.lineJoin = 'round';
  g.beginPath();
  for (let i = 0; i < bvp.length; i++) {
    const x = (i / (bvp.length - 1)) * w, y = h / 2 - (bvp[i] / m) * (h / 2 - 6);
    if (i) g.lineTo(x, y); else g.moveTo(x, y);
  }
  g.stroke();
}

export interface SpecColumn { t: number; emission: Float64Array; track: number; status: string }

/** Fused harmonic-summed spectrum over time with the tracker output. */
export function drawSpectrogram(c: HTMLCanvasElement, cols: SpecColumn[], rows: HrRow[], spanS: number, grid: [number, number]) {
  const { g, w, h } = fitCanvas(c);
  const padL = 34, padB = 16;
  const pw = w - padL, ph = h - padB;
  const css = getComputedStyle(c);
  g.clearRect(0, 0, w, h);
  g.fillStyle = css.getPropertyValue('--chart-bg') || '#11161c';
  g.fillRect(padL, 0, pw, ph);
  const [lo, hi] = grid;
  const yOf = (bpm: number) => ph - ((bpm - lo) / (hi - lo)) * ph;
  if (cols.length) {
    const tEnd = cols[cols.length - 1].t, t0 = tEnd - spanS;
    const xOf = (t: number) => padL + ((t - t0) / spanS) * pw;
    const colW = Math.max(1, pw / spanS + 0.5);
    for (const col of cols) {
      if (col.t < t0) continue;
      const n = col.emission.length;
      const x = xOf(col.t) - colW;
      for (let i = 0; i < n; i += 2) {
        const v = Math.max(col.emission[i], col.emission[Math.min(n - 1, i + 1)]);
        if (v < 0.04) continue;
        g.fillStyle = `hsla(${40 - 40 * v}, 95%, ${18 + 50 * v}%, ${0.25 + 0.75 * v})`;
        const y0 = ph - ((i + 2) / n) * ph;
        g.fillRect(x, y0, colW, (2 / n) * ph + 1);
      }
    }
    // current (non-lagged) Viterbi state
    g.strokeStyle = 'rgba(255,255,255,0.55)';
    g.setLineDash([3, 3]);
    g.lineWidth = 1;
    g.beginPath();
    let started = false;
    for (const col of cols) {
      if (col.t < t0 || !Number.isFinite(col.track) || col.status === 'NO_FACE' || col.status === 'LOW_COVERAGE') { started = false; continue; }
      if (started) g.lineTo(xOf(col.t), yOf(col.track)); else g.moveTo(xOf(col.t), yOf(col.track));
      started = true;
    }
    g.stroke();
    g.setLineDash([]);
    // decided (fixed-lag) HR track
    g.strokeStyle = '#ff5a6e';
    g.lineWidth = 2.5;
    g.beginPath();
    started = false;
    for (const r of rows) {
      if (r.time_s < t0 || !Number.isFinite(r.hr_bpm)) { started = false; continue; }
      if (started) g.lineTo(xOf(r.time_s), yOf(r.hr_bpm)); else g.moveTo(xOf(r.time_s), yOf(r.hr_bpm));
      started = true;
    }
    g.stroke();
  }
  g.fillStyle = css.getPropertyValue('--muted') || '#8a94a6';
  g.font = '11px system-ui, sans-serif';
  g.textAlign = 'right';
  for (const bpm of [60, 90, 120, 150, 180]) {
    const y = yOf(bpm);
    g.fillText(String(bpm), padL - 6, Math.max(10, y + 4));
    g.fillStyle = 'rgba(128,128,128,0.18)';
    g.fillRect(padL, y, pw, 1);
    g.fillStyle = css.getPropertyValue('--muted') || '#8a94a6';
  }
  g.textAlign = 'left';
  g.fillText(`last ${spanS} s`, padL + 4, h - 3);
}
