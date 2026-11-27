// viterbiTrack.ts — physiology-constrained HR path over window spectra, a port
// of src/signal/viterbiHR.m (main repo) plus an online fixed-lag variant for
// the CP3 core's 1 Hz output.
//
//   states      HR grid 42:0.5:180 bpm
//   emission    log(h + 1e-12), h = interp1(60 f, P, grid, 'linear', 0) clipped
//               at 0 and normalised to unit sum (no spectrum = flat)
//   transition  h -> h' over the step dt allowed only if
//               -maxDownBpm dt <= h' - h <= maxUpBpm dt, score -lambda |h' - h|
//   output      path state refined by parabolic interpolation on exp(emission)
//               (+-0.5 bin); conf = emission mass within +-3 bpm of the state
//
// Defaults: symmetric 4.99 bpm per second (operator: no change of 5 bpm or
// more between readings), lambda 0.1 per bpm, lag 3 s. The online tracker
// derives the transition limits from the actual step between outputs. A
// fixed-lag decision is not tied to the previous decision by a transition
// (each decision backtracks from the current best state), so the online
// output is additionally limited to +-maxUpBpm dt from the previous output
// ("clamped"; rare).
export const VITERBI = {
  grid: [42, 180, 0.5] as readonly [number, number, number],
  maxDownBpm: 4.99,   // per second of output step
  maxUpBpm: 4.99,     // per second of output step
  lambda: 0.1,        // per bpm
  lagS: 3,
  stepS: 1,           // nominal output step (lag in steps = lagS / stepS)
  gapResetS: 10,      // restart the path after more than this without a spectrum
  confHalfBpm: 3,
} as const;

export interface ViterbiParams {
  grid: readonly [number, number, number];
  maxDownBpm: number; maxUpBpm: number; lambda: number; lagS: number; stepS: number; gapResetS: number; confHalfBpm: number;
}

/** grid(1):grid(3):grid(2) (exact for the 0.5 bpm default). */
export const viterbiGrid = (p: Pick<ViterbiParams, 'grid'> = VITERBI) =>
  Float64Array.from({ length: Math.floor((p.grid[1] - p.grid[0]) / p.grid[2] + 1e-9) + 1 }, (_, i) => p.grid[0] + i * p.grid[2]);

/** MATLAB emission row: log(normalised interp1(60 f, P, grid, 'linear', 0) + 1e-12); null = P of ones (flat). */
export function viterbiEmission(Pin: ArrayLike<number> | null, fHz: ArrayLike<number>, grid: Float64Array): Float64Array {
  const nS = grid.length, h = new Float64Array(nS);
  const P = Pin ?? new Float64Array(fHz.length).fill(1);
  if (fHz.length >= 2) {
    const n = fHz.length;
    let j = 0;
    for (let s = 0; s < nS; s++) {
      const q = grid[s];
      const x0 = 60 * fHz[0], x1 = 60 * fHz[n - 1];
      if (!(q >= x0 && q <= x1)) continue;
      while (j < n - 2 && 60 * fHz[j + 1] < q) j++;
      const a = 60 * fHz[j], b = 60 * fHz[j + 1];
      const v = q === b ? P[j + 1] : P[j] + ((q - a) / (b - a)) * (P[j + 1] - P[j]);
      h[s] = Math.max(Number.isFinite(v) ? v : 0, 0);
    }
  }
  let sum = 0;
  for (const v of h) sum += v;
  const d = Math.max(sum, Number.EPSILON);
  return Float64Array.from(h, (v) => Math.log(v / d + 1e-12));
}

/** Parabolic refinement (clamped to +-0.5 bin) and +-confHalfBpm mass, as viterbiHR.m. */
function refine(E: Float64Array, grid: Float64Array, k: number, step: number, confHalf: number): { hr: number; conf: number } {
  let hr = grid[k];
  const nS = grid.length;
  if (k > 0 && k < nS - 1) {
    const a = Math.exp(E[k - 1]), b = Math.exp(E[k]), c = Math.exp(E[k + 1]), den = a - 2 * b + c;
    if (den !== 0) hr += step * Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / den));
  }
  let conf = 0;
  for (let s = 0; s < nS; s++) if (Math.abs(grid[s] - grid[k]) <= confHalf) conf += Math.exp(E[s]);
  return { hr, conf };
}

/** One Viterbi recursion: best predecessor per state (first on ties, like MATLAB max). */
function recurse(score: Float64Array, E: Float64Array, grid: Float64Array, dt: number, p: ViterbiParams) {
  const nS = grid.length, next = new Float64Array(nS), back = new Int32Array(nS);
  const dn = -p.maxDownBpm * dt, up = p.maxUpBpm * dt;
  for (let j = 0; j < nS; j++) {
    let best = -Infinity, arg = 0;
    for (let i = 0; i < nS; i++) {
      const d = grid[j] - grid[i];
      if (d < dn || d > up) continue;
      const v = score[i] - p.lambda * Math.abs(d);
      if (v > best) { best = v; arg = i; }
    }
    next[j] = best + E[j];
    back[j] = arg;
  }
  return { next, back };
}

const argmax = (x: Float64Array) => { let k = 0; for (let i = 1; i < x.length; i++) if (x[i] > x[k]) k = i; return k; };

/** Offline full Viterbi (viterbiHR.m). P rows: spectra on fHz (null = flat). */
export function viterbiHR(P: (ArrayLike<number> | null)[], fHz: ArrayLike<number>, opts: Partial<ViterbiParams> & { step?: number } = {}) {
  const p: ViterbiParams = { ...VITERBI, ...opts };
  const dt = opts.step ?? p.stepS;
  const grid = viterbiGrid(p), nW = P.length;
  const E = P.map((row) => viterbiEmission(row, fHz, grid));
  let score = Float64Array.from(E[0]);
  const backs: Int32Array[] = [new Int32Array(0)];
  for (let w = 1; w < nW; w++) {
    const r = recurse(score, E[w], grid, dt, p);
    score = r.next; backs.push(r.back);
  }
  const path = new Int32Array(nW);
  path[nW - 1] = argmax(score);
  for (let w = nW - 1; w >= 1; w--) path[w - 1] = backs[w][path[w]];
  const out = Array.from(path, (k, w) => refine(E[w], grid, k, p.grid[2], p.confHalfBpm));
  return { hr: out.map((o) => o.hr), conf: out.map((o) => o.conf), path };
}

export interface ViterbiDecision {
  index: number;      // step index (0-based, counting update() calls since the last reset)
  tS: number;         // time of that step
  hr: number;         // bpm
  conf: number;
  clamped: boolean;   // output limited to +-maxUpBpm dt from the previous output
}

/** Online fixed-lag Viterbi: update() once per output step, decisions lag `lagS` behind. */
export class ViterbiTracker {
  readonly p: ViterbiParams;
  readonly grid: Float64Array;
  readonly lagSteps: number;
  private score: Float64Array | null = null;
  private steps: { tS: number; E: Float64Array; back: Int32Array | null }[] = [];
  private n = 0;            // steps since reset
  private decidedUpTo = -1; // last decided step index
  private lastT = NaN;
  private lastValidT = -Infinity;
  private lastOut: { hr: number; tS: number } | null = null;

  constructor(params: Partial<ViterbiParams> = {}) {
    this.p = { ...VITERBI, ...params };
    this.grid = viterbiGrid(this.p);
    this.lagSteps = Math.round(this.p.lagS / this.p.stepS);
  }

  reset() {
    this.score = null;
    this.steps = [];
    this.n = 0;
    this.decidedUpTo = -1;
    this.lastT = NaN;
    this.lastValidT = -Infinity;
    this.lastOut = null;
  }

  get stepCount() { return this.n; }
  /** Time of the newest / oldest step still held (NaN when empty). */
  get lastStepT() { return this.steps.length ? this.steps[this.steps.length - 1].tS : NaN; }
  get firstStepT() { return this.steps.length ? this.steps[0].tS : NaN; }

  /** Current (not lagged) best state, bpm. */
  current(): number {
    return this.score ? this.grid[argmax(this.score)] : NaN;
  }

  /**
   * Feed one window (null = no spectrum: flat emission). Returns the decisions
   * made by this step: normally the step `lagSteps` back; after a gap longer
   * than gapResetS the remaining path is flushed and the tracker restarts.
   */
  update(spectrum: ArrayLike<number> | null, fHz: ArrayLike<number>, tS: number): ViterbiDecision[] {
    const out: ViterbiDecision[] = [];
    if (!spectrum && this.score && tS - this.lastValidT > this.p.gapResetS + 1e-9) {
      out.push(...this.flush());
      this.reset();
    }
    if (!spectrum && !this.score) return out; // nothing to track yet
    const dt = Number.isFinite(this.lastT) && tS > this.lastT ? tS - this.lastT : this.p.stepS;
    this.lastT = tS;
    if (spectrum) this.lastValidT = tS;
    const E = viterbiEmission(spectrum, fHz, this.grid);
    if (!this.score) {
      this.score = Float64Array.from(E);
      this.steps.push({ tS, E, back: null });
    } else {
      const r = recurse(this.score, E, this.grid, dt, this.p);
      const mx = Math.max(...r.next);
      for (let i = 0; i < r.next.length; i++) r.next[i] -= mx; // keep scores bounded
      this.score = r.next;
      this.steps.push({ tS, E, back: r.back });
    }
    this.n++;
    const target = this.n - 1 - this.lagSteps;
    if (target > this.decidedUpTo) out.push(...this.decide(target));
    // keep lag + 1 steps of back-pointers
    while (this.steps.length > this.lagSteps + 1) this.steps.shift();
    return out;
  }

  /** Decide every step still inside the lag (end of input). */
  flush(): ViterbiDecision[] {
    return this.score ? this.decide(this.n - 1) : [];
  }

  private decide(upTo: number): ViterbiDecision[] {
    const first = this.n - this.steps.length; // step index of steps[0]
    const path = new Int32Array(this.steps.length);
    path[path.length - 1] = argmax(this.score!);
    for (let k = path.length - 1; k >= 1; k--) path[k - 1] = this.steps[k].back![path[k]];
    const out: ViterbiDecision[] = [];
    for (let idx = Math.max(this.decidedUpTo + 1, first); idx <= upTo; idx++) {
      const s = this.steps[idx - first];
      let { hr, conf } = refine(s.E, this.grid, path[idx - first], this.p.grid[2], this.p.confHalfBpm);
      let clamped = false;
      if (this.lastOut) {
        const dt = s.tS - this.lastOut.tS;
        const lo = this.lastOut.hr - this.p.maxDownBpm * dt, hi = this.lastOut.hr + this.p.maxUpBpm * dt;
        if (hr < lo || hr > hi) { hr = Math.min(hi, Math.max(lo, hr)); clamped = true; }
      }
      this.lastOut = { hr, tS: s.tS };
      out.push({ index: idx, tS: s.tS, hr, conf, clamped });
    }
    this.decidedUpTo = Math.max(this.decidedUpTo, upTo);
    return out;
  }
}
