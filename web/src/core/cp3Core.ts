// cp3Core.ts — the frozen Round-1 signal chain "CP3" (config/round1_freeze.json in
// the main repo), a line-by-line port of src/pipeline/snrFusionHR.m:
//
//   per patch: Kalman denoise on the native frame times (kalman_denoise.m,
//   R_k ~ 1/npix, -3 dB at min(4 Hz, 0.4 fs)) -> linear resample to 30 Hz on
//   the real timestamps (edges: nearest) -> POS 1.6 s (std N-1) -> zero-phase
//   Butterworth band-pass ORDER 3, 0.7-3 Hz (MATLAB butter + filtfilt on b/a,
//   no detrend, no z-score) -> 10 s windows / 1 s step: a window needs >= 90 %
//   of its samples, a patch >= 90 % valid samples in the window and >= 50 %
//   valid frames overall -> symmetric Hann periodogram (nfft 8192), unit
//   in-band power -> linear SNR (+-0.1 Hz around the in-band peak, +-0.2 Hz
//   around its harmonic, over the rest of the band) -> fused = sum SNR_i P_i /
//   sum(P_i in band) -> HR = 60 * argmax (no interpolation).
//
// cp3Analyse() is the offline function (whole recording, tests/cp3Parity.test.ts
// compares it with MATLAB). Cp3Core is the online SignalCore: once per second it
// runs cp3Analyse on the newest window_s + pad_s seconds of TraceFrames and
// evaluates the single window that ends now (centre T - 5 s). No Viterbi, no
// harmonic summation, no motion suppression, no sub-patches (CP3 flags).
//
// Parameters are the CP3 freeze, not shared/config.default.json (which still
// holds the interim values: order 4, detrend, nfft 16384, ...).
//
// Online HR tracking of the 1 Hz output on the fused in-band spectrum of each
// window (cp3Analyse() stays argmax = frozen CP3), so a reading never changes
// by 5 bpm or more between consecutive outputs:
//   'viterbi' (default)  fixed-lag Viterbi, symmetric 4.99 bpm per second of
//                        step, lambda 0.1 / bpm, lag 3 s (viterbiTrack.ts, port
//                        of src/signal/viterbiHR.m)
//   'troika'             TROIKA peak tracking, no lag (troikaTrack.ts, port of
//                        src/track/troikaTrack.m)
//   'off'                raw argmax (exact MATLAB snrFusionHR window HRs)
import { CONFIG, MASKED_RULE_CODES, PATCHES } from '../config';
import { fftInPlace, hann, std, zscore } from './dsp';
import { kalmanDenoise } from './kalman';
import { hrGrid, sampleOnGrid } from './tracker';
import { TroikaTracker } from './troikaTrack';
import { ViterbiTracker } from './viterbiTrack';

export type TrackerKind = 'viterbi' | 'troika' | 'off';
import type { CoreTick, HrRow, PatchQuality, SignalCore, Status, TraceFrame } from './types';

/** CP3 frozen parameters (round1_freeze.json, snrFusionHR.m defaults). */
export const CP3 = {
  fs: 30,
  band: [0.7, 3.0] as [number, number],
  order: 3,
  win: 10,
  step: 1,
  posWin: 1.6,
  nfftMin: 8192,
  kalmanCutoff: 4,
  kalmanNyqFrac: 0.8, // -3 dB at min(4, 0.8 * fs / 2) = min(4, 0.4 fs)
  peakHalfHz: 0.1,
  harmHalfHz: 0.2,
  winFrac: 0.9,
  patchWinFrac: 0.9,
  patchAllFrac: 0.5,
  /** online only: history before the analysed window (Kalman/POS/filter settle) */
  padS: 5,
} as const;

// ------------------------------------------------------------ Butterworth (b, a)
type Cx = [number, number];
const cmul = (a: Cx, b: Cx): Cx => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cdiv = (a: Cx, b: Cx): Cx => {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};
const csqrt = (a: Cx): Cx => {
  const r = Math.hypot(a[0], a[1]);
  return [Math.sqrt((r + a[0]) / 2), Math.sign(a[1] || 1) * Math.sqrt(Math.max(0, (r - a[0]) / 2))];
};
/** MATLAB poly(r): monic polynomial (descending powers) with roots r. */
function poly(r: Cx[]): Cx[] {
  let c: Cx[] = [[1, 0]];
  for (const z of r) {
    const n: Cx[] = [...c, [0, 0]];
    for (let i = 1; i < n.length; i++) {
      const p = cmul(c[i - 1], z);
      n[i] = [n[i][0] - p[0], n[i][1] - p[1]];
    }
    c = n;
  }
  return c;
}

/**
 * MATLAB [b, a] = butter(n, band / (fs / 2), 'bandpass'): analog prototype ->
 * lp2bp -> bilinear (prewarped, fs = 2) -> a = poly(poles); b = poly(zeros)
 * scaled so that H = 1 at the band centre (MATLAB buttnum).
 */
export function butterBandpassBA(n: number, band: readonly [number, number], fs: number): { b: number[]; a: number[] } {
  const u1 = 4 * Math.tan((Math.PI * (band[0] / (fs / 2))) / 2);
  const u2 = 4 * Math.tan((Math.PI * (band[1] / (fs / 2))) / 2);
  const Bw = u2 - u1, Wo = Math.sqrt(u1 * u2);
  const poles: Cx[] = [];
  for (let k = 1; k <= n; k++) {
    const th = (Math.PI * (2 * k + n - 1)) / (2 * n);
    const pb: Cx = [(Math.cos(th) * Bw) / 2, (Math.sin(th) * Bw) / 2];
    const d = csqrt([pb[0] * pb[0] - pb[1] * pb[1] - Wo * Wo, 2 * pb[0] * pb[1]]);
    for (const s of [[pb[0] + d[0], pb[1] + d[1]], [pb[0] - d[0], pb[1] - d[1]]] as Cx[]) {
      poles.push(cdiv([4 + s[0], s[1]], [4 - s[0], -s[1]]));
    }
  }
  const a = poly(poles).map((c) => c[0]);
  const b0 = poly([...Array(n).fill([1, 0]), ...Array(n).fill([-1, 0])] as Cx[]).map((c) => c[0]);
  const w = 2 * Math.atan2(Wo, 4);
  let ka: Cx = [0, 0], kb: Cx = [0, 0];
  for (let k = 0; k < a.length; k++) {
    const e: Cx = [Math.cos(w * k), -Math.sin(w * k)];
    ka = [ka[0] + e[0] * a[k], ka[1] + e[1] * a[k]];
    kb = [kb[0] + e[0] * b0[k], kb[1] + e[1] * b0[k]];
  }
  const g = cdiv(ka, kb);
  return { b: b0.map((v) => v * g[0]), a };
}

/** Direct-form II transposed filter (MATLAB filter) with initial state z (modified in place). */
function lfilter(b: number[], a: number[], x: Float64Array, z: Float64Array): Float64Array {
  const m = z.length, y = new Float64Array(x.length);
  for (let n = 0; n < x.length; n++) {
    const xn = x[n], yn = b[0] * xn + z[0];
    for (let i = 0; i < m - 1; i++) z[i] = b[i + 1] * xn + z[i + 1] - a[i + 1] * yn;
    z[m - 1] = b[m] * xn - a[m] * yn;
    y[n] = yn;
  }
  return y;
}

/** filtfilt initial conditions (MATLAB getCoeffsAndInitialConditions, b/a form). */
function filtfiltZi(b: number[], a: number[]): Float64Array {
  const nf = a.length, m = nf - 1;
  const M = Array.from({ length: m }, () => new Float64Array(m));
  M[0][0] = 1 + a[1];
  for (let i = 1; i < m; i++) { M[i][0] = a[i + 1]; M[i][i] = 1; M[i - 1][i] = -1; }
  const r = Float64Array.from({ length: m }, (_, i) => b[i + 1] - b[0] * a[i + 1]);
  for (let c = 0; c < m; c++) { // Gaussian elimination, partial pivoting
    let p = c;
    for (let i = c + 1; i < m; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
    [M[c], M[p]] = [M[p], M[c]];
    [r[c], r[p]] = [r[p], r[c]];
    for (let i = c + 1; i < m; i++) {
      const f = M[i][c] / M[c][c];
      if (!f) continue;
      for (let j = c; j < m; j++) M[i][j] -= f * M[c][j];
      r[i] -= f * r[c];
    }
  }
  const zi = new Float64Array(m);
  for (let i = m - 1; i >= 0; i--) {
    let s = r[i];
    for (let j = i + 1; j < m; j++) s -= M[i][j] * zi[j];
    zi[i] = s / M[i][i];
  }
  return zi;
}

/**
 * MATLAB filtfilt(b, a, x): odd reflection of nfact = 3 (nfilt - 1) samples at
 * both ends, forward filter with zi * xt(1), reverse filter with zi * y(end).
 * Same edge handling as MATLAB (no padding difference).
 */
export function filtfiltBA(b: number[], a: number[], x: Float64Array): Float64Array {
  const nf = a.length, nfact = 3 * (nf - 1), N = x.length;
  if (N <= nfact) throw new Error(`filtfiltBA: signal too short (${N} <= ${nfact})`);
  const zi = filtfiltZi(b, a);
  const xt = new Float64Array(N + 2 * nfact);
  for (let i = 0; i < nfact; i++) {
    xt[i] = 2 * x[0] - x[nfact - i];
    xt[N + nfact + i] = 2 * x[N - 1] - x[N - 2 - i];
  }
  xt.set(x, nfact);
  const y = lfilter(b, a, xt, zi.map((v) => v * xt[0])).reverse();
  const y2 = lfilter(b, a, y, zi.map((v) => v * y[0])).reverse();
  return y2.slice(nfact, nfact + N);
}

// --------------------------------------------------------------- helpers
/** MATLAB colon a:d:b (Moler's colonop: symmetric from both ends). */
export function colon(a: number, d: number, b: number): Float64Array {
  if (!(d > 0) || b < a) return new Float64Array(0);
  const tol = 2 * Number.EPSILON * Math.max(Math.abs(a), Math.abs(b));
  let n = Math.round((b - a) / d);
  if (a + n * d - b > tol) n--;
  let c = a + n * d;
  if (c - b > -tol) c = b;
  const v = new Float64Array(n + 1);
  for (let k = 0; k <= Math.floor(n / 2); k++) { v[k] = a + k * d; v[n - k] = c - k * d; }
  if (n % 2 === 0) v[n / 2] = (a + c) / 2;
  return v;
}

/** MATLAB median of finite diffs -> native frame rate. */
function medianOf(a: number[]): number {
  const s = Float64Array.from(a).sort();
  const n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2])) : NaN;
}

/** interp1(tx, vx, tq, 'linear') with NaN (or `outside`) outside [tx(1), tx(end)]; tx strictly increasing. */
function interpLinear(tx: number[], vx: number[], tq: Float64Array, outside = NaN): Float64Array {
  const out = new Float64Array(tq.length).fill(outside);
  const n = tx.length;
  if (!n) return out;
  let j = 0;
  for (let i = 0; i < tq.length; i++) {
    const q = tq[i];
    if (q < tx[0] || q > tx[n - 1]) continue;
    if (n === 1) { out[i] = vx[0]; continue; }
    while (j < n - 2 && tx[j + 1] < q) j++;
    const w = (q - tx[j]) / (tx[j + 1] - tx[j]);
    out[i] = q === tx[j + 1] ? vx[j + 1] : vx[j] + w * (vx[j + 1] - vx[j]);
  }
  return out;
}

/** fillmissing(x, 'nearest') for the leading / trailing NaN runs left by interp1. */
function fillEdges(x: Float64Array): Float64Array {
  let i0 = 0, i1 = x.length - 1;
  while (i0 < x.length && !Number.isFinite(x[i0])) i0++;
  while (i1 >= 0 && !Number.isFinite(x[i1])) i1--;
  if (i0 > i1) return x;
  for (let i = 0; i < i0; i++) x[i] = x[i0];
  for (let i = i1 + 1; i < x.length; i++) x[i] = x[i1];
  return x;
}

/** POS (Wang 2017) exactly as posPulse in snrFusionHR.m: 1.6 s overlap-add, std N-1, mu <= 0 skipped. */
export function posCp3(r: Float64Array, g: Float64Array, b: Float64Array, fs: number): Float64Array {
  const N = r.length, l = Math.round(CP3.posWin * fs), h = new Float64Array(N);
  const s1 = new Float64Array(l), s2 = new Float64Array(l), pp = new Float64Array(l);
  for (let n = 0; n + l <= N; n++) {
    let mr = 0, mg = 0, mb = 0;
    for (let k = n; k < n + l; k++) { mr += r[k]; mg += g[k]; mb += b[k]; }
    mr /= l; mg /= l; mb /= l;
    if (mr <= 0 || mg <= 0 || mb <= 0) continue;
    for (let k = 0; k < l; k++) {
      const cr = r[n + k] / mr, cg = g[n + k] / mg, cb = b[n + k] / mb;
      s1[k] = cg - cb;
      s2[k] = -2 * cr + cg + cb;
    }
    const sd2 = std(s2);
    if (sd2 === 0) continue;
    const ratio = std(s1) / sd2;
    let m = 0;
    for (let k = 0; k < l; k++) { pp[k] = s1[k] + ratio * s2[k]; m += pp[k]; }
    m /= l;
    for (let k = 0; k < l; k++) h[n + k] += pp[k] - m;
  }
  return h;
}

// ------------------------------------------------------------- analysis
export interface Cp3PatchWin {
  okFrac: number;      // share of the window's 30 Hz samples with a valid frame nearby
  used: boolean;       // qualified (>= 90 % valid in the window, >= 50 % overall, finite, in-band power > 0)
  snr: number;         // linear SNR (0 if unused)
  f0: number;          // in-band peak, Hz (NaN if unused)
}

export interface Cp3Window {
  tc: number;
  hr: number;              // bpm, NaN when no patch qualifies or the window lacks samples
  enoughSamples: boolean;  // >= 90 % of the window's samples on the 30 Hz grid
  patches: Cp3PatchWin[];
  fused: Float64Array | null; // fused in-band spectrum (bins fb)
  confidence: number;      // fused power within +-0.1 Hz of the peak / fused in-band power
  bvp: Float64Array | null;   // SNR-weighted, sign-aligned band-passed POS of the window
}

export interface Cp3Result {
  tc: Float64Array;
  hr: Float64Array;
  fb: Float64Array;        // in-band frequency bins, Hz
  df: number;
  windows: Cp3Window[];
}

/**
 * Offline CP3 (snrFusionHR.m). t: N frame times (s). X[r]: N x 3 mean RGB of
 * patch r (NaN rows = invalid). npix[r]: N pixel counts (or null). centers:
 * optional window centres (default tu(1)+5 : 1 : tu(end)-5).
 */
export function cp3Analyse(
  t: number[], X: number[][][], npix: (number[] | null)[], centers?: number[],
): Cp3Result {
  const { fs, band, win, step } = CP3;
  const R = X.length, N = t.length;
  const L = Math.round(win * fs);
  const nfft = Math.max(CP3.nfftMin, 2 ** Math.ceil(Math.log2(L)));
  const df = fs / nfft;
  const fbArr: number[] = [];
  let k0 = -1;
  for (let k = 0; k <= nfft / 2; k++) {
    const f = (k * fs) / nfft;
    if (f >= band[0] && f <= band[1]) { if (k0 < 0) k0 = k; fbArr.push(f); }
  }
  const fb = Float64Array.from(fbArr), nb = fb.length;
  const empty = (tc: Float64Array): Cp3Result => ({
    tc, hr: new Float64Array(tc.length).fill(NaN), fb, df,
    windows: Array.from(tc, (c) => ({ tc: c, hr: NaN, enoughSamples: false, patches: [], fused: null, confidence: 0, bvp: null })),
  });

  const tFinite: number[] = [];
  for (let i = 1; i < N; i++) if (Number.isFinite(t[i]) && Number.isFinite(t[i - 1])) tFinite.push(t[i] - t[i - 1]);
  const fsNative = 1 / medianOf(tFinite);
  const valid = X.map((x) => x.map((row, i) => Number.isFinite(t[i]) && row.every(Number.isFinite)));
  const anyValid = t.map((_, i) => valid.some((v) => v[i]));
  const nAny = anyValid.filter(Boolean).length;
  if (!(nAny >= win * fs * 0.9 * (fsNative / fs)) || !Number.isFinite(fsNative)) return empty(Float64Array.from(centers ?? []));
  const tv = t.filter((_, i) => anyValid[i]);
  const tu = colon(tv[0], 1 / fs, tv[tv.length - 1] + 1e-9);
  const { b, a } = butterBandpassBA(CP3.order, band, fs);

  const tIdx = t.map((_, i) => i).filter((i) => Number.isFinite(t[i]));
  const H: (Float64Array | null)[] = [], okU: (Uint8Array | null)[] = [];
  for (let r = 0; r < R; r++) {
    const v = valid[r];
    const nv = v.filter(Boolean).length;
    if (nv < 0.5 * nAny) { H.push(null); okU.push(null); continue; }
    const w = npix[r] ? npix[r]!.map((q) => Math.max(q, 1)) : null;
    const y = kalmanDenoise(X[r], fsNative, w, CP3.kalmanCutoff, CP3.kalmanNyqFrac);
    const tv2: number[] = [];
    for (let i = 0; i < N; i++) if (v[i]) tv2.push(t[i]);
    const ch = [0, 1, 2].map((c) => {
      const vals: number[] = [];
      for (let i = 0; i < N; i++) if (v[i]) vals.push(y[i][c]);
      return fillEdges(interpLinear(tv2, vals, tu));
    });
    try {
      H.push(filtfiltBA(b, a, posCp3(ch[0], ch[1], ch[2], fs)));
    } catch {
      H.push(null); okU.push(null); continue;
    }
    const ok = interpLinear(tIdx.map((i) => t[i]), tIdx.map((i) => (v[i] ? 1 : 0)), tu, 0);
    okU.push(Uint8Array.from(ok, (q) => (q > 0.5 ? 1 : 0)));
  }

  const tc = centers ? Float64Array.from(centers) : colon(tu[0] + win / 2, step, tu[tu.length - 1] - win / 2 + 1e-9);
  const hr = new Float64Array(tc.length).fill(NaN);
  const windows: Cp3Window[] = [];
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let i = 0; i < tc.length; i++) {
    const lo = tc[i] - win / 2, hi = tc[i] + win / 2;
    const idx: number[] = [];
    for (let j = 0; j < tu.length; j++) if (tu[j] >= lo && tu[j] < hi) idx.push(j);
    const nS = idx.length;
    const W: Cp3Window = { tc: tc[i], hr: NaN, enoughSamples: nS >= 0.9 * L, patches: [], fused: null, confidence: 0, bvp: null };
    windows.push(W);
    const hw = hann(Math.max(nS, 1));
    let fused: Float64Array | null = null;
    const segs: { y: Float64Array; w: number }[] = [];
    for (let r = 0; r < R; r++) {
      const pw: Cp3PatchWin = { okFrac: 0, used: false, snr: 0, f0: NaN };
      W.patches.push(pw);
      const hr_ = H[r], ok = okU[r];
      if (!hr_ || !ok || !nS) continue;
      let okc = 0;
      for (const j of idx) okc += ok[j];
      pw.okFrac = okc / nS;
      if (!W.enoughSamples || pw.okFrac < CP3.patchWinFrac) continue;
      const y = Float64Array.from(idx, (j) => hr_[j]);
      let m = 0;
      for (const v of y) m += v;
      m /= nS;
      if (!y.every(Number.isFinite)) continue;
      for (let j = 0; j < nS; j++) y[j] -= m;
      re.fill(0); im.fill(0);
      for (let j = 0; j < nS; j++) re[j] = y[j] * hw[j];
      fftInPlace(re, im);
      const Pb = new Float64Array(nb);
      let tot = 0;
      for (let k = 0; k < nb; k++) { const q = k0 + k; Pb[k] = re[q] * re[q] + im[q] * im[q]; tot += Pb[k]; }
      if (!(tot > 0)) continue;
      let kmax = 0;
      for (let k = 1; k < nb; k++) if (Pb[k] > Pb[kmax]) kmax = k;
      const f0 = fb[kmax];
      let sig = 0, noise = 0;
      for (let k = 0; k < nb; k++) {
        if (Math.abs(fb[k] - f0) <= CP3.peakHalfHz || Math.abs(fb[k] - 2 * f0) <= CP3.harmHalfHz) sig += Pb[k];
        else noise += Pb[k];
      }
      const snr = sig / Math.max(noise, 1e-30);
      Object.assign(pw, { used: true, snr, f0 });
      if (!fused) fused = new Float64Array(nb);
      for (let k = 0; k < nb; k++) fused[k] += (snr * Pb[k]) / tot;
      segs.push({ y, w: snr });
    }
    if (!fused) continue;
    let kmax = 0;
    for (let k = 1; k < nb; k++) if (fused[k] > fused[kmax]) kmax = k;
    hr[i] = W.hr = 60 * fb[kmax];
    W.fused = fused;
    let num = 0, den = 0;
    for (let k = 0; k < nb; k++) { den += fused[k]; if (Math.abs(fb[k] - fb[kmax]) <= CP3.peakHalfHz) num += fused[k]; }
    W.confidence = den > 0 ? Math.max(0, Math.min(1, num / den)) : 0;
    W.bvp = fuseBvp(segs);
  }
  return { tc, hr, fb, df, windows };
}

/** SNR-weighted sum of the patch window signals, signs aligned with the strongest patch, z-scored. */
function fuseBvp(segs: { y: Float64Array; w: number }[]): Float64Array | null {
  if (!segs.length) return null;
  const ref = segs.reduce((p, q) => (q.w > p.w ? q : p)).y;
  const out = new Float64Array(ref.length);
  for (const { y, w } of segs) {
    let c = 0;
    for (let i = 0; i < y.length; i++) c += y[i] * ref[i];
    const s = c < 0 ? -w : w;
    for (let i = 0; i < y.length; i++) out[i] += s * y[i];
  }
  return zscore(out);
}

// ------------------------------------------------------------ online core
const MASK_IDS = new Set(PATCHES.filter((p) => MASKED_RULE_CODES.includes(p.code)).map((p) => p.id));

/**
 * Online CP3 SignalCore. Every step_s it analyses the window [T - 10 s, T)
 * with cp3Analyse on the frames of the last 10 + padS seconds and emits a row
 * immediately (no tracker lag).
 *
 * Status (same codes as the interim core; never OK without an HR):
 *   WARMUP        less than 10 s since the first frame (face seen recently)
 *   NO_FACE       no tracked face in the last no_face_s
 *   LOW_COVERAGE  window lacks >= 90 % of its samples or no patch qualifies
 *   LOW_SNR       a patch qualifies but none has linear SNR > 1 (0 dB: the
 *                 peak + harmonic carry less power than the rest of the band;
 *                 the same 0 dB floor the interim core's weights use)
 *   MASKED        lower-face occlusion rule (README 4.3.5, as interim); HR is
 *                 still CP3's (occluded patches drop out by the 90 % rule)
 *   OK            otherwise
 * Confidence = share of the fused in-band power within +-0.1 Hz of the fused
 * peak (in [0, 1]; a clean sinusoid gives ~0.9, a flat spectrum ~0.1).
 *
 * HR tracking (constructor option `tracker`, default 'viterbi'): the reported
 * HR is the tracked peak of the same fused spectrum and the confidence is taken
 * around it. With 'viterbi' the row of window T is released at T + 3 s (fixed
 * lag; finish() flushes the rest); 'troika' and 'off' emit without lag. The
 * tracker is fed every window after warm-up (NO_FACE / no spectrum = gap: more
 * than 10 s without a spectrum restarts it) and is reset with the core (new
 * target). Status codes are those of the window; never OK without an HR.
 */
interface PendingRow {
  T: number;
  status: Status;
  used: PatchQuality[];
  fused: Float64Array | null;
  fb: Float64Array;
  fed: boolean;      // given to the Viterbi tracker (else released without a decision)
  hr: number;        // decided HR (NaN until decided)
  decided: boolean;
}

/** Share of the fused in-band power within +-0.1 Hz of hr (W.confidence around a given HR). */
function confidenceAt(fused: Float64Array, fb: Float64Array, hr: number): number {
  let num = 0, den = 0;
  for (let k = 0; k < fb.length; k++) { den += fused[k]; if (Math.abs(60 * fb[k] - hr) <= 60 * CP3.peakHalfHz) num += fused[k]; }
  return den > 0 ? Math.max(0, Math.min(1, num / den)) : 0;
}

const makeRow = (T: number, status: Status, hr = NaN, conf = 0, used: PatchQuality[] = []): HrRow => ({
  time_s: T, hr_bpm: hr, confidence: conf, status, n_patches_used: used.length,
  top_patches: [...used].sort((p, q) => q.weight - p.weight).map((p) => p.parent),
});

export class Cp3Core implements SignalCore {
  private frames: TraceFrame[] = [];
  private t0: number | null = null;
  private nextTick = 0;
  private queue: PendingRow[] = [];
  readonly grid = hrGrid(CONFIG);
  readonly tracker: TrackerKind;
  readonly troika: TroikaTracker | null;
  readonly viterbi: ViterbiTracker | null;

  constructor(opts: { tracker?: TrackerKind } = {}) {
    this.tracker = opts.tracker ?? 'viterbi';
    this.troika = this.tracker === 'troika' ? new TroikaTracker() : null;
    this.viterbi = this.tracker === 'viterbi' ? new ViterbiTracker({ stepS: CP3.step }) : null;
  }

  /** Output lag of the HR rows (s). */
  get lagS(): number {
    return this.viterbi ? this.viterbi.lagSteps * CP3.step : 0;
  }

  reset() {
    this.frames = [];
    this.t0 = null;
    this.queue = [];
    this.troika?.reset();
    this.viterbi?.reset();
  }

  push(frame: TraceFrame): CoreTick[] {
    if (this.frames.length && frame.t_s <= this.frames[this.frames.length - 1].t_s) return [];
    if (this.t0 === null) {
      this.t0 = frame.t_s;
      this.nextTick = frame.t_s + CP3.step;
    }
    this.frames.push(frame);
    const ticks: CoreTick[] = [];
    while (frame.t_s >= this.nextTick) {
      ticks.push(this.tick(this.nextTick));
      this.nextTick += CP3.step;
    }
    const keep = this.nextTick - (CP3.win + CP3.padS + 2 * CP3.step);
    let drop = 0;
    while (drop < this.frames.length - 1 && this.frames[drop].t_s < keep) drop++;
    if (drop) this.frames.splice(0, drop);
    return ticks;
  }

  finish(): HrRow[] {
    if (!this.viterbi) return []; // rows are emitted without lag
    this.applyDecisions(this.viterbi.flush());
    for (const q of this.queue) q.decided = true;
    return this.release();
  }

  private applyDecisions(ds: { tS: number; hr: number }[]) {
    for (const d of ds) {
      const q = this.queue.find((e) => e.fed && e.T === d.tS);
      if (q) { q.hr = d.hr; q.decided = true; }
    }
  }

  /** Rows of the decided windows at the head of the queue, oldest first. */
  private release(): HrRow[] {
    const out: HrRow[] = [];
    while (this.queue.length && this.queue[0].decided) {
      const q = this.queue.shift()!;
      const okStatus = q.status === 'OK' || q.status === 'MASKED';
      const ok = okStatus && Number.isFinite(q.hr) && !!q.fused;
      const status: Status = okStatus && !ok ? 'LOW_COVERAGE' : q.status;
      out.push(makeRow(q.T, status, ok ? q.hr : NaN, ok ? confidenceAt(q.fused!, q.fb, q.hr) : 0, ok ? q.used : []));
    }
    return out;
  }

  /** Viterbi path: queue the window, feed the tracker, attach the released rows. */
  private viterbiRows(tick: CoreTick, status: Status, fused: Float64Array | null, fb: Float64Array, used: PatchQuality[], feed: boolean) {
    const v = this.viterbi!;
    const q: PendingRow = { T: tick.t_s, status, used, fused, fb, fed: false, hr: NaN, decided: false };
    this.queue.push(q);
    if (feed) {
      const ds = v.update(fused, fb, tick.t_s);
      q.fed = v.lastStepT === tick.t_s;
      this.applyDecisions(ds);
    }
    if (!q.fed) q.decided = true;
    // windows the tracker no longer holds (dropped by a restart) cannot be decided any more
    for (const e of this.queue) if (e.fed && !e.decided && !(e.T >= v.firstStepT)) e.decided = true;
    tick.trackBpm = v.current();
    const rows = this.release();
    if (rows.length) { tick.rows = rows; tick.row = rows[rows.length - 1]; }
  }

  private tick(T: number): CoreTick {
    const elapsed = T - (this.t0 ?? T);
    const frames = this.frames.filter((f) => f.t_s <= T && f.t_s > T - CP3.win - CP3.padS);
    const faceRecent = frames.some((f) => f.face_status > 0 && f.t_s > T - CONFIG.no_face_s);
    const tick: CoreTick = {
      t_s: T, status: 'WARMUP', fs_proc: CP3.fs, patches: [], gridBpm: this.grid,
      emission: new Float64Array(this.grid.length), trackBpm: NaN, bvp: new Float64Array(0),
    };
    if (elapsed < CP3.win - 1e-9 || !faceRecent) {
      tick.status = faceRecent ? 'WARMUP' : 'NO_FACE';
      const noFace = tick.status === 'NO_FACE';
      // no spectrum: a gap for the tracker (more than 10 s without one restarts it)
      if (this.troika && noFace) tick.troika = { rawBpm: NaN, ...this.troika.update(null, [], T) };
      if (this.viterbi) this.viterbiRows(tick, tick.status, null, new Float64Array(0), [], noFace);
      else tick.row = makeRow(T, tick.status);
      return tick;
    }

    const t = frames.map((f) => f.t_s);
    const X: number[][][] = [], np: number[][] = [];
    for (const def of PATCHES) {
      const x: number[][] = [], n: number[] = [];
      for (const f of frames) {
        const p = f.patches.find((q) => q.id === def.id);
        x.push(p && p.valid && f.face_status > 0 ? [...p.rgb_mean] : [NaN, NaN, NaN]);
        n.push(p ? p.npix : 0);
      }
      X.push(x); np.push(n);
    }
    const res = cp3Analyse(t, X, np, [T - CP3.win / 2]);
    const W = res.windows[0];
    // window HR: raw argmax (MATLAB snrFusionHR) or the TROIKA-tracked peak of the same fused spectrum
    let hr = W.hr, conf = W.confidence;
    if (this.troika) {
      const o = this.troika.update(W.fused, res.fb, T);
      tick.troika = { rawBpm: W.hr, ...o };
      if (W.fused) { hr = o.hr; conf = confidenceAt(W.fused, res.fb, hr); }
    }

    const snrSum = W.patches.reduce((s, p) => s + (p.used ? p.snr : 0), 0);
    tick.patches = PATCHES.map((def, r) => {
      const p = W.patches[r] ?? { okFrac: 0, used: false, snr: 0, f0: NaN };
      return {
        id: def.id, parent: def.id, sub: false, coverage: p.okFrac,
        snr_db: p.used ? 10 * Math.log10(Math.max(p.snr, 1e-12)) : -Infinity,
        peak_bpm: 60 * p.f0, weight: p.used && snrSum > 0 ? p.snr / snrSum : 0, used: p.used,
      };
    });
    const used = tick.patches.filter((p) => p.used);

    const recent = frames.filter((f) => f.t_s > T - CP3.step && f.face_status > 0);
    const maskedFrames = recent.filter((f) => {
      const ps = f.patches.filter((p) => MASK_IDS.has(p.id));
      return ps.length > 0 && ps.filter((p) => !p.valid && p.skin_ratio < CONFIG.occlusion_off).length / ps.length >= CONFIG.masked_fraction;
    }).length;
    const masked = recent.length > 0 && maskedFrames / recent.length > 0.5;

    let status: Status;
    if (!Number.isFinite(hr)) status = 'LOW_COVERAGE';
    else if (!used.some((p) => p.snr_db > 0)) status = 'LOW_SNR';
    else status = masked ? 'MASKED' : 'OK';
    tick.status = status;

    if (W.fused) {
      const full = new Float64Array(Math.round(CP3.fs / res.df / 2) + 1);
      const k0 = Math.round(res.fb[0] / res.df);
      full.set(W.fused, k0);
      const Hg = sampleOnGrid(full, res.df, this.grid);
      let mx = 0;
      for (const v of Hg) mx = Math.max(mx, v);
      if (mx > 0) for (let i = 0; i < Hg.length; i++) tick.emission[i] = Hg[i] / mx;
      tick.trackBpm = hr;
    }
    if (W.bvp) tick.bvp = W.bvp;
    if (this.viterbi) {
      this.viterbiRows(tick, status, W.fused, res.fb, used, true);
      return tick;
    }
    const ok = status === 'OK' || status === 'MASKED';
    tick.row = makeRow(T, status, ok ? hr : NaN, ok ? conf : 0, ok ? used : []);
    return tick;
  }
}
