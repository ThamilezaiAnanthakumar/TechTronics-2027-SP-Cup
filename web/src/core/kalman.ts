// kalman.ts — port of src/signal/kalman_denoise.m (Thamil).
// Per channel: level = random walk (Q), measurement = level + noise R_k,
// R_k = R0 * median(npix) / npix_k. R0 = var(diff(x)) / 2. Q is set so the
// steady-state filter (at median npix) is a first-order low-pass whose -3 dB
// point is min(cutoff, nyqFrac * fs / 2). README 4.4 fixes nyqFrac = 0.8;
// kalman_denoise.m uses 0.45*fs (identical at 10 fps, where both give 4 Hz).
import { median } from './dsp';

/** Steady-state gain alpha of y += alpha (x - y) with |H| = 1/sqrt(2) at fc. */
export function kalmanAlpha(fs: number, cutoffHz: number, nyqFrac: number): number {
  const fc = Math.min(cutoffHz, (nyqFrac * fs) / 2);
  const w = (2 * Math.PI * fc) / fs;
  const c = 4 - 2 * Math.cos(w);
  const beta = (c - Math.sqrt(c * c - 4)) / 2;
  return 1 - beta;
}

function variance(a: number[]): number {
  const n = a.length;
  if (n < 2) return 0;
  let m = 0;
  for (const v of a) m += v;
  m /= n;
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return s / (n - 1);
}

/**
 * x: N rows x C channels (NaN rows allowed and kept NaN).
 * npix: optional per-row pixel counts behind each mean.
 */
export function kalmanDenoise(
  x: number[][],
  fs: number,
  npix: number[] | null,
  cutoffHz: number,
  nyqFrac: number,
): number[][] {
  const N = x.length;
  const Cn = N ? x[0].length : 0;
  const alpha = kalmanAlpha(fs, cutoffHz, nyqFrac);
  let scale: number[];
  if (!npix) scale = new Array(N).fill(1);
  else {
    const med = median(npix.filter((v) => v > 0));
    scale = npix.map((v) => med / Math.max(v, 1));
  }
  const y = x.map(() => new Array<number>(Cn).fill(NaN));
  for (let ch = 0; ch < Cn; ch++) {
    const d: number[] = [];
    for (let k = 1; k < N; k++) {
      const v = x[k][ch] - x[k - 1][ch];
      if (Number.isFinite(v)) d.push(v);
    }
    const R0 = variance(d) / 2;
    if (!(R0 > 0)) {
      for (let k = 0; k < N; k++) y[k][ch] = x[k][ch];
      continue;
    }
    const Q = (R0 * alpha * alpha) / (1 - alpha);
    let s = NaN, P = NaN;
    for (let k = 0; k < N; k++) {
      if (!Number.isNaN(s)) P += Q;
      const v = x[k][ch];
      if (!Number.isFinite(v)) continue;
      if (Number.isNaN(s)) {
        s = v;
        P = R0 * scale[k];
      } else {
        const R = R0 * scale[k];
        const G = P / (P + R);
        s += G * (v - s);
        P *= 1 - G;
      }
      y[k][ch] = s;
    }
  }
  return y;
}
