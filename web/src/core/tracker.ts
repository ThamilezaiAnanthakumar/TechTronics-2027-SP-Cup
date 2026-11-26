// tracker.ts — README 4.6-4.8: per-patch SNR, harmonic summation and the
// online fixed-lag Viterbi HR tracker with physiological rate limits.
import type { FvConfig } from '../config';
import { parabolicOffset } from './dsp';

export function hrGrid(cfg: Pick<FvConfig, 'hr_grid_bpm'>): Float64Array {
  const [lo, hi, step] = cfg.hr_grid_bpm;
  const n = Math.round((hi - lo) / step) + 1;
  return Float64Array.from({ length: n }, (_, i) => lo + i * step);
}

/** In-band bin range [k0, k1] for a one-sided spectrum with bin width df. */
export function bandBins(band: readonly [number, number], df: number, nBins: number): [number, number] {
  const k0 = Math.max(0, Math.ceil(band[0] / df - 1e-9));
  const k1 = Math.min(nBins - 1, Math.floor(band[1] / df + 1e-9));
  return [k0, k1];
}

/**
 * SNR in dB (de Haan & Jeanne 2013 style, README 4.6): in-band power within
 * +-halfwidth of the dominant in-band peak f0 and of 2 f0, over the remaining
 * in-band power. Returns the peak frequency too.
 */
export function spectralSnr(
  P: Float64Array, df: number, band: readonly [number, number], halfwidthHz: number,
): { snrDb: number; f0: number } {
  const [k0, k1] = bandBins(band, df, P.length);
  let kmax = k0;
  for (let k = k0; k <= k1; k++) if (P[k] > P[kmax]) kmax = k;
  const f0 = (kmax + parabolicOffset(P, kmax)) * df;
  let sig = 0, noise = 0;
  for (let k = k0; k <= k1; k++) {
    const f = k * df;
    if (Math.abs(f - f0) <= halfwidthHz || Math.abs(f - 2 * f0) <= halfwidthHz) sig += P[k];
    else noise += P[k];
  }
  const snrDb = noise > 0 ? 10 * Math.log10(sig / noise) : sig > 0 ? 60 : -60;
  return { snrDb, f0 };
}

/** H(f) = P(f) + alpha P(2f), the 2f term only where 2f < fLimit (README 4.8.1). */
export function harmonicSum(P: Float64Array, df: number, alpha: number, fLimit: number): Float64Array {
  const H = Float64Array.from(P);
  for (let k = 0; k < P.length; k++) {
    const k2 = 2 * k;
    if (k2 < P.length && k2 * df < fLimit) H[k] += alpha * P[k2];
  }
  return H;
}

/** Linear interpolation of a uniformly sampled spectrum onto the HR grid (bpm). */
export function sampleOnGrid(H: Float64Array, df: number, gridBpm: Float64Array): Float64Array {
  return Float64Array.from(gridBpm, (bpm) => {
    const x = bpm / 60 / df;
    const i = Math.floor(x);
    if (i < 0) return H[0];
    if (i >= H.length - 1) return H[H.length - 1];
    const w = x - i;
    return H[i] * (1 - w) + H[i + 1] * w;
  });
}

/**
 * Online confidence (README 4.8.5): normalised peak prominence of the path's
 * peak = 1 - (largest other local maximum outside +-halfwidth) / (path peak).
 */
export function peakProminence(Hg: Float64Array, idx: number, halfwidthBins: number): number {
  let h1 = 0;
  for (let i = Math.max(0, idx - halfwidthBins); i <= Math.min(Hg.length - 1, idx + halfwidthBins); i++) h1 = Math.max(h1, Hg[i]);
  if (!(h1 > 0)) return 0;
  let h2 = 0;
  for (let i = 0; i < Hg.length; i++) {
    if (Math.abs(i - idx) <= halfwidthBins) continue;
    const left = i === 0 ? -Infinity : Hg[i - 1], right = i === Hg.length - 1 ? -Infinity : Hg[i + 1];
    if (Hg[i] >= left && Hg[i] >= right) h2 = Math.max(h2, Hg[i]);
  }
  return Math.max(0, Math.min(1, 1 - h2 / h1));
}

/** Path bin refined by parabolic interpolation on the grid emission (README 4.8.4). */
export function refineBpm(Hg: Float64Array, gridBpm: Float64Array, idx: number): number {
  const step = gridBpm.length > 1 ? gridBpm[1] - gridBpm[0] : 0;
  return gridBpm[idx] + parabolicOffset(Hg, idx) * step;
}

export interface ViterbiStep {
  current: number;                            // best state now (not lagged)
  decided: { step: number; state: number } | null; // fixed-lag decision for step - lag
}

/**
 * Online fixed-lag Viterbi over the HR grid (README 4.8.3).
 * Transition h -> h' over dt allowed only if down*dt <= h'-h <= up*dt, cost
 * lambda |h'-h|. Emission = log(H/sum(H) + eps); null = no information.
 */
export class FixedLagViterbi {
  private delta: Float64Array | null = null;
  private back: Int32Array[] = [];
  private nSteps = 0;
  private lo: number;
  private hi: number;
  private cost: Float64Array;

  constructor(
    readonly grid: Float64Array,
    private cfg: Pick<FvConfig, 'viterbi_max_down_bpm_s' | 'viterbi_max_up_bpm_s' | 'viterbi_lambda_per_bpm' | 'emission_eps'>,
    readonly lagSteps: number,
    dtS = 1,
  ) {
    const step = grid[1] - grid[0];
    // predecessor offsets i - j (bins) for target j: i in [j - up/step, j - down/step]
    this.lo = Math.floor((-cfg.viterbi_max_up_bpm_s * dtS) / step + 1e-9);
    this.hi = Math.ceil((-cfg.viterbi_max_down_bpm_s * dtS) / step - 1e-9);
    this.cost = Float64Array.from({ length: this.hi - this.lo + 1 }, (_, k) => cfg.viterbi_lambda_per_bpm * Math.abs((this.lo + k) * step));
  }

  reset() {
    this.delta = null;
    this.back = [];
    this.nSteps = 0;
  }

  get steps() { return this.nSteps; }

  private emission(H: Float64Array | null): Float64Array {
    const n = this.grid.length;
    if (!H) return new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.max(H[i], 0);
    if (!(s > 0)) return new Float64Array(n);
    return Float64Array.from(H, (v) => Math.log(Math.max(v, 0) / s + this.cfg.emission_eps));
  }

  step(H: Float64Array | null): ViterbiStep {
    const e = this.emission(H);
    const n = this.grid.length;
    const next = new Float64Array(n);
    const bp = new Int32Array(n);
    if (!this.delta) {
      next.set(e);
      for (let j = 0; j < n; j++) bp[j] = j;
    } else {
      const d = this.delta;
      for (let j = 0; j < n; j++) {
        let best = -Infinity, arg = j;
        for (let o = this.lo; o <= this.hi; o++) {
          const i = j + o;
          if (i < 0 || i >= n) continue;
          const v = d[i] - this.cost[o - this.lo];
          if (v > best) { best = v; arg = i; }
        }
        next[j] = best + e[j];
        bp[j] = arg;
      }
    }
    let mx = -Infinity, cur = 0;
    for (let j = 0; j < n; j++) if (next[j] > mx) { mx = next[j]; cur = j; }
    for (let j = 0; j < n; j++) next[j] -= mx; // keep scores bounded
    this.delta = next;
    this.back.push(bp);
    if (this.back.length > this.lagSteps + 1) this.back.shift();
    const stepIdx = this.nSteps++;
    let decided: ViterbiStep['decided'] = null;
    if (stepIdx >= this.lagSteps) {
      let s = cur;
      for (let k = this.back.length - 1; k > this.back.length - 1 - this.lagSteps; k--) s = this.back[k][s];
      decided = { step: stepIdx - this.lagSteps, state: s };
    }
    return { current: cur, decided };
  }

  /** Decide all steps that are still inside the lag (end of input). */
  flush(): { step: number; state: number }[] {
    if (!this.delta) return [];
    const undecided = Math.min(this.lagSteps, this.nSteps);
    let s = 0;
    for (let j = 1; j < this.delta.length; j++) if (this.delta[j] > this.delta[s]) s = j;
    const out: { step: number; state: number }[] = [];
    // the newest step's state is s; walk back through the stored pointers
    for (let k = 0; k < undecided; k++) {
      out.push({ step: this.nSteps - 1 - k, state: s });
      s = this.back[this.back.length - 1 - k][s];
    }
    return out.reverse();
  }
}
