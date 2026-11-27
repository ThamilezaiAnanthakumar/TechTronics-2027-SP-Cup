// troikaTrack.ts — online TROIKA-style spectral peak tracking, a port of
// src/track/troikaTrack.m (main repo). Zhang, Pi & Liu, TROIKA, IEEE TBME
// 62(2), 2015, Section III-D (Eqs. 13-18), with every parameter in physical
// units instead of FFT bins at 125 Hz:
//
//   init       highest spectral peak of the first valid window
//   selection  R0 = prev +- searchBpm (+- searchLostBpm while lost), R1 = 2x
//              that range (first harmonic); up to 3 highest local peaks
//              >= peakRel x max(R0) in each. Harmonic pair (|f1 - 2 f0| <=
//              harmTolBpm) -> that fundamental (Eq. 13); else the candidate
//              (R0 peaks, halved R1 peaks) nearest to prev (Eq. 14); none ->
//              prev (Eq. 15)
//   verify     |est - prev| >= jumpBpm * step -> prev +- stepBpm * step (Eq. 16)
//   lost       est == prev for lostWindows windows -> prev +- 2 stepBpm * step
//              by the sign of a cubic-fit trend over the last trendWindows
//              windows predicted one step ahead, when |pred - prev| >= trendBpm
//              (Eqs. 17-18)
//   gaps       re-initialise after more than gapResetS without a valid window
//
// step = seconds between consecutive update() calls (the actual output step),
// so the per-window limits follow the real output rate.

/** MATLAB troikaTrack.m defaults (physical units; rates are per second of output step). */
export const TROIKA = {
  stepS: 1,            // assumed step for the first update (MATLAB Step)
  jumpBpm: 5,          // per second: a reading never changes by this much or more
  stepBpm: 2,          // per second: clamp step (tau)
  searchBpm: 15,
  searchLostBpm: 19,
  peakRel: 0.3,
  harmTolBpm: 3,
  lostWindows: 3,
  trendWindows: 20,
  trendBpm: 3,
  gapResetS: 10,
} as const;

export type TroikaParams = { -readonly [K in keyof typeof TROIKA]: number };
export type TroikaAction = 'init' | 'harmonic' | 'nearest' | 'hold' | 'clamp' | 'trend' | 'gap';

export interface TroikaOut {
  hr: number;          // bpm, NaN for a gap / before the first valid window
  action: TroikaAction;
}

/** Up to three highest local maxima of p inside [i0, i1) (contiguous mask) with p >= thr. */
function topPeaks(p: Float64Array, idx: number[], thr: number): number[] {
  if (idx.length < 3) return [];
  const pk: number[] = [];
  for (let j = 1; j < idx.length - 1; j++) {
    const q = p[idx[j]];
    if (q > p[idx[j - 1]] && q >= p[idx[j + 1]] && q >= thr) pk.push(idx[j]);
  }
  pk.sort((a, b) => p[b] - p[a]); // stable, like MATLAB sort 'descend'
  return pk.slice(0, 3);
}

/** MATLAB polyval(polyfit(1:n, h, 3), n + 1): least squares via Householder QR. */
export function cubicPredictNext(h: number[]): number {
  const n = h.length, m = 4;
  const A = h.map((_, i) => { const x = i + 1; return [x * x * x, x * x, x, 1]; });
  const y = [...h];
  for (let k = 0; k < m; k++) {
    let nrm = 0;
    for (let i = k; i < n; i++) nrm += A[i][k] * A[i][k];
    nrm = Math.sqrt(nrm);
    if (nrm === 0) continue;
    const alpha = A[k][k] > 0 ? -nrm : nrm;
    const v = new Array<number>(n).fill(0);
    for (let i = k; i < n; i++) v[i] = A[i][k];
    v[k] -= alpha;
    let vv = 0;
    for (let i = k; i < n; i++) vv += v[i] * v[i];
    if (vv === 0) continue;
    for (let j = k; j < m; j++) {
      let s = 0;
      for (let i = k; i < n; i++) s += v[i] * A[i][j];
      s = (2 * s) / vv;
      for (let i = k; i < n; i++) A[i][j] -= s * v[i];
    }
    let s = 0;
    for (let i = k; i < n; i++) s += v[i] * y[i];
    s = (2 * s) / vv;
    for (let i = k; i < n; i++) y[i] -= s * v[i];
  }
  const c = new Array<number>(m).fill(0);
  for (let k = m - 1; k >= 0; k--) {
    let s = y[k];
    for (let j = k + 1; j < m; j++) s -= A[k][j] * c[j];
    c[k] = s / A[k][k];
  }
  const x = n + 1;
  return ((c[0] * x + c[1]) * x + c[2]) * x + c[3];
}

/** Stateful online tracker: one update() per output window. */
export class TroikaTracker {
  readonly p: TroikaParams;
  private prev = NaN;
  private nSame = 0;
  private lastValidT = -Infinity;
  private lastT = NaN;
  private hist: number[] = []; // last trendWindows window estimates (NaN = no estimate)

  constructor(params: Partial<TroikaParams> = {}) {
    this.p = { ...TROIKA, ...params };
  }

  reset() {
    this.prev = NaN;
    this.nSame = 0;
    this.lastValidT = -Infinity;
    this.lastT = NaN;
    this.hist = [];
  }

  /**
   * spectrum: window spectrum on freqsHz (null / all non-finite = no estimate
   * this window); tS: window time (s). Returns the tracked HR and the action.
   */
  update(spectrum: ArrayLike<number> | null, freqsHz: ArrayLike<number>, tS: number): TroikaOut {
    const o = this.p;
    const dt = Number.isFinite(this.lastT) && tS > this.lastT ? tS - this.lastT : o.stepS;
    this.lastT = tS;
    const n = freqsHz.length;
    const p = new Float64Array(n);
    let anyFinite = false;
    if (spectrum) {
      for (let k = 0; k < n; k++) {
        const v = spectrum[k];
        if (Number.isFinite(v)) { p[k] = v; anyFinite = true; }
      }
    }
    if (!anyFinite) {
      if (Number.isFinite(this.prev) && tS - this.lastValidT > o.gapResetS + 1e-9) { this.prev = NaN; this.nSame = 0; }
      this.pushHist(NaN);
      return { hr: NaN, action: 'gap' };
    }
    const bpm = Float64Array.from({ length: n }, (_, k) => 60 * freqsHz[k]);
    const jump = o.jumpBpm * dt, tau = o.stepBpm * dt;
    let est: number, action: TroikaAction;
    if (!Number.isFinite(this.prev)) { // (re-)initialisation
      let k = 0;
      for (let i = 1; i < n; i++) if (p[i] > p[k]) k = i;
      est = bpm[k]; action = 'init';
    } else {
      const prev = this.prev;
      const half = this.nSame >= o.lostWindows ? o.searchLostBpm : o.searchBpm;
      const r0: number[] = [], r1: number[] = [];
      let mx = -Infinity;
      for (let k = 0; k < n; k++) {
        if (Math.abs(bpm[k] - prev) <= half) { r0.push(k); mx = Math.max(mx, p[k]); }
        if (bpm[k] >= 2 * (prev - half) && bpm[k] <= 2 * (prev + half)) r1.push(k);
      }
      const thr = r0.length ? o.peakRel * mx : NaN;
      const c0 = topPeaks(p, r0, thr), c1 = topPeaks(p, r1, thr);
      est = NaN; action = 'hold';
      for (const i of c0) { // Eq. 13: harmonic pair
        if (c1.some((j) => Math.abs(bpm[j] - 2 * bpm[i]) <= o.harmTolBpm)) { est = bpm[i]; action = 'harmonic'; break; }
      }
      if (Number.isNaN(est)) {
        const cand = [...c0.map((i) => bpm[i]), ...c1.map((j) => bpm[j] / 2)];
        if (cand.length) { // Eq. 14: nearest to the previous estimate (first on ties, like MATLAB min)
          let best = 0;
          for (let i = 1; i < cand.length; i++) if (Math.abs(cand[i] - prev) < Math.abs(cand[best] - prev)) best = i;
          est = cand[best]; action = 'nearest';
        } else {
          est = prev; action = 'hold'; // Eq. 15
        }
      }
      if (Math.abs(est - prev) >= jump) { // Eq. 16: verification / clamp
        est = prev + Math.sign(est - prev) * tau; action = 'clamp';
      }
      this.nSame = est === prev ? this.nSame + 1 : 0;
      if (this.nSame >= o.lostWindows) { // Eqs. 17-18: trend step when stuck
        const h = this.hist.filter(Number.isFinite);
        if (h.length >= 4) {
          const d = cubicPredictNext(h) - prev;
          if (d >= o.trendBpm) { est = prev + 2 * tau; action = 'trend'; }
          else if (d <= -o.trendBpm) { est = prev - 2 * tau; action = 'trend'; }
        }
      }
    }
    this.pushHist(est);
    this.prev = est;
    this.lastValidT = tS;
    return { hr: est, action };
  }

  private pushHist(v: number) {
    this.hist.push(v);
    if (this.hist.length > this.p.trendWindows) this.hist.shift();
  }
}
