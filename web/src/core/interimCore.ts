// interimCore.ts — interim TypeScript signal core (README Section 16
// "Interim rule"). Implements SignalCore so fvcore-WASM can replace it.
//
// Per 1 Hz tick: per patch Kalman (R_k ~ 1/npix) on the raw irregular
// samples -> uniform resample (real timestamps, gaps > 0.5 s stay NaN) ->
// POS/CHROM -> smoothness-priors detrend -> Butterworth BPF -> z-score ->
// Hann spectrum -> SNR / coherence / coverage weights -> top-k spectral
// fusion -> harmonic summation -> online fixed-lag Viterbi -> HR row.
import { CONFIG, MASKED_RULE_CODES, PATCHES, type FvConfig } from '../config';
import { bandpass, chrom, detrendSmoothnessPriors, median, pos, powerSpectrum, zscore, type Rgb } from './dsp';
import { kalmanDenoise } from './kalman';
import { LOCAL_CONFIG, type LocalConfig } from '../localConfig';
import { fillNaN, largeMotionFrames, motionSpectrum, type MotionSpectrum } from './motion';
import {
  FixedLagViterbi, bandBins, harmonicSum, hrGrid, peakProminence, refineBpm, sampleOnGrid, spectralSnr,
} from './tracker';
import type { CoreTick, HrRow, PatchQuality, SignalCore, Status, TraceFrame } from './types';

interface PendingStep {
  time_s: number;
  status: Status;
  Hg: Float64Array | null;
  nUsed: number;
  top: number[];
}

interface PatchResult {
  id: number;
  group: number;
  coverage: number;
  snrDb: number;
  peakBpm: number;
  Pn: Float64Array | null; // full one-sided spectrum, unit in-band energy
  seg: Float64Array | null;
  weight: number;
}

const MASK_IDS = new Set(PATCHES.filter((p) => MASKED_RULE_CODES.includes(p.code)).map((p) => p.id));

/** Linear resampling of irregular samples onto tg; gaps > maxGap stay NaN. */
export function resampleUniform(t: number[], v: number[], tg: Float64Array, maxGap: number, edge: number): Float64Array {
  const out = new Float64Array(tg.length).fill(NaN);
  const ti: number[] = [], vi: number[] = [];
  for (let i = 0; i < t.length; i++) if (Number.isFinite(v[i])) { ti.push(t[i]); vi.push(v[i]); }
  if (!ti.length) return out;
  let j = 0;
  for (let i = 0; i < tg.length; i++) {
    const q = tg[i];
    if (q < ti[0]) { if (ti[0] - q <= edge) out[i] = vi[0]; continue; }
    if (q > ti[ti.length - 1]) { if (q - ti[ti.length - 1] <= edge) out[i] = vi[vi.length - 1]; continue; }
    while (j < ti.length - 2 && ti[j + 1] < q) j++;
    const ta = ti[j], tb = ti[Math.min(j + 1, ti.length - 1)];
    if (tb === ta) { out[i] = vi[j]; continue; }
    if (tb - ta > maxGap) continue;
    const w = (q - ta) / (tb - ta);
    out[i] = vi[j] * (1 - w) + vi[j + 1] * w;
  }
  return out;
}

export class InterimCore implements SignalCore {
  private frames: TraceFrame[] = [];
  private t0: number | null = null;
  private nextTick = 0;
  private viterbi: FixedLagViterbi;
  private pending = new Map<number, PendingStep>();
  readonly grid: Float64Array;

  constructor(private cfg: FvConfig = CONFIG, private local: LocalConfig = LOCAL_CONFIG) {
    this.grid = hrGrid(cfg);
    this.viterbi = new FixedLagViterbi(this.grid, cfg, Math.round(cfg.viterbi_lag_s / cfg.step_s), cfg.step_s);
  }

  reset() {
    this.frames = [];
    this.t0 = null;
    this.viterbi.reset();
    this.pending.clear();
  }

  push(frame: TraceFrame): CoreTick[] {
    if (this.frames.length && frame.t_s <= this.frames[this.frames.length - 1].t_s) return [];
    if (this.t0 === null) {
      this.t0 = frame.t_s;
      this.nextTick = frame.t_s + this.cfg.step_s;
    }
    this.frames.push(frame);
    const ticks: CoreTick[] = [];
    while (frame.t_s >= this.nextTick) {
      ticks.push(this.tick(this.nextTick));
      this.nextTick += this.cfg.step_s;
    }
    const keep = this.nextTick - (this.cfg.window_s + this.cfg.pad_s + 2 * this.cfg.step_s);
    let drop = 0;
    while (drop < this.frames.length - 1 && this.frames[drop].t_s < keep) drop++;
    if (drop) this.frames.splice(0, drop);
    return ticks;
  }

  finish(): HrRow[] {
    return this.viterbi.flush().map((d) => this.makeRow(d.step, d.state)).filter((r): r is HrRow => !!r);
  }

  private makeRow(step: number, state: number): HrRow | null {
    const p = this.pending.get(step);
    if (!p) return null;
    this.pending.delete(step);
    const ok = p.status === 'OK' || p.status === 'MASKED';
    const halfBins = Math.round(this.cfg.confidence_halfwidth_bpm / this.cfg.hr_grid_bpm[2]);
    return {
      time_s: p.time_s,
      hr_bpm: ok && p.Hg ? refineBpm(p.Hg, this.grid, state) : NaN,
      confidence: ok && p.Hg ? peakProminence(p.Hg, state, halfBins) : 0,
      status: p.status,
      n_patches_used: p.nUsed,
      top_patches: p.top,
    };
  }

  private tick(T: number): CoreTick {
    const cfg = this.cfg;
    const elapsed = T - (this.t0 ?? T);
    const frames = this.frames.filter((f) => f.t_s <= T && f.t_s > T - cfg.window_s - cfg.pad_s);
    const faceRecent = frames.some((f) => f.face_status > 0 && f.t_s > T - cfg.no_face_s);
    const empty: CoreTick = {
      t_s: T, status: 'WARMUP', fs_proc: 0, patches: [], gridBpm: this.grid,
      emission: new Float64Array(this.grid.length), trackBpm: NaN, bvp: new Float64Array(0),
    };

    if (elapsed < cfg.window_s - 1e-9) {
      const status: Status = faceRecent ? 'WARMUP' : 'NO_FACE';
      empty.status = status;
      empty.row = { time_s: T, hr_bpm: NaN, confidence: 0, status, n_patches_used: 0, top_patches: [] };
      return empty;
    }

    // measured frame rate over the analysed span (all frames, face or not)
    const span = frames.length > 1 ? frames[frames.length - 1].t_s - frames[0].t_s : 0;
    const fsMeasured = span > 0 ? (frames.length - 1) / span : 0;
    const fsProc = Math.min(cfg.fs_proc_max, Math.round(fsMeasured));
    const fsOk = fsProc > 2 * cfg.band_hz[1] && frames.length > 10;

    let results: PatchResult[] = [];
    let masked = false;
    let motion: MotionSpectrum | null = null;
    if (faceRecent && fsOk) {
      const padAvail = Math.max(0, Math.min(cfg.pad_s, elapsed - cfg.window_s));
      const n = Math.floor((cfg.window_s + padAvail) * fsProc);
      const Lw = Math.round(cfg.window_s * fsProc);
      const tg = Float64Array.from({ length: n }, (_, i) => T - (n - 1 - i) / fsProc);
      const tf = frames.map((f) => f.t_s);
      const reject = largeMotionFrames(frames, this.local.motion_reject_step);
      motion = motionSpectrum(frames, tf, tg, Lw, fsProc, cfg, this.local);
      const pAtt = this.local.motion_patch_level ? motion.atten : null;
      results = PATCHES.map((def) => this.analysePatch(def.id, def.groupIndex, frames, tf, tg, Lw, fsMeasured, fsProc, reject, pAtt));

      // MASKED (README 4.3.5): in the last second, >= masked_fraction of the
      // group-C + malar patches invalid in most face frames.
      const recent = frames.filter((f) => f.t_s > T - cfg.step_s && f.face_status > 0);
      const maskedFrames = recent.filter((f) => {
        const ps = f.patches.filter((p) => MASK_IDS.has(p.id));
        return ps.length > 0 && ps.filter((p) => !p.valid).length / ps.length >= cfg.masked_fraction;
      }).length;
      masked = recent.length > 0 && maskedFrames / recent.length > 0.5;
    }

    // coherence + weights (README 4.6)
    const covOk = results.filter((r) => r.Pn && r.coverage >= cfg.coverage_min);
    const med = median(covOk.map((r) => r.peakBpm));
    for (const r of covOk) {
      const d = Math.abs(r.peakBpm - med);
      const coh = d <= cfg.coherence_tol_bpm ? 1 : Math.exp(-(d - cfg.coherence_tol_bpm) / cfg.coherence_decay_bpm);
      const snrLin = Math.pow(10, r.snrDb / 10);
      r.weight = r.coverage * Math.pow(Math.max(snrLin - 1, 0), cfg.weight_power) * coh;
      if (masked && r.group === 2) r.weight = 0;
    }
    const used = results.filter((r) => r.weight > 0).sort((a, b) => b.weight - a.weight).slice(0, cfg.top_k);
    const wSum = used.reduce((s, r) => s + r.weight, 0);

    let status: Status;
    if (!faceRecent) status = 'NO_FACE';
    else if (!fsOk || !covOk.length) status = 'LOW_COVERAGE';
    else if (!used.length) status = 'LOW_SNR';
    else status = masked ? 'MASKED' : 'OK';

    // fused spectrum (README 4.7) and harmonic summation (4.8.1)
    let Hg: Float64Array | null = null;
    let bvp = new Float64Array(0);
    if (used.length) {
      const len = used[0].Pn!.length;
      const Pf = new Float64Array(len);
      for (const r of used) for (let k = 0; k < len; k++) Pf[k] += (r.weight / wSum) * r.Pn![k];
      const df = fsProc / ((len - 1) * 2);
      // README 4.8.2 motion suppression (web-local options, see localConfig.ts)
      const att = !this.local.motion_patch_level && motion?.atten && motion.atten.length === len ? motion.atten : null;
      if (att && this.local.motion_before_harmonic) for (let k = 0; k < len; k++) Pf[k] *= att[k];
      const H = harmonicSum(Pf, df, cfg.harmonic_alpha, (cfg.harmonic_nyquist_frac * fsProc) / 2);
      if (att && !this.local.motion_before_harmonic) for (let k = 0; k < len; k++) H[k] *= att[k];
      Hg = sampleOnGrid(H, df, this.grid);
      bvp = this.fuseBvp(used, wSum);
    } else {
      const best = covOk[0];
      if (best?.seg) bvp = best.seg;
    }

    // Viterbi carries on through uninformative windows (emission = uniform)
    const informative = status === 'OK' || status === 'MASKED';
    const stepIdx = this.viterbi.steps;
    this.pending.set(stepIdx, {
      time_s: T, status, Hg: informative ? Hg : null,
      nUsed: informative ? used.length : 0, top: informative ? used.map((r) => r.id) : [],
    });
    const vs = this.viterbi.step(informative ? Hg : null);
    const row = vs.decided ? this.makeRow(vs.decided.step, vs.decided.state) ?? undefined : undefined;

    const emission = new Float64Array(this.grid.length);
    if (Hg && informative) {
      let mx = 0;
      for (const v of Hg) mx = Math.max(mx, v);
      if (mx > 0) for (let i = 0; i < Hg.length; i++) emission[i] = Hg[i] / mx;
    }
    const patches: PatchQuality[] = results.map((r) => ({
      id: r.id, coverage: r.coverage, snr_db: r.snrDb, peak_bpm: r.peakBpm,
      weight: used.includes(r) ? r.weight / wSum : 0, used: used.includes(r),
    }));
    return {
      t_s: T, status: elapsed < cfg.window_s + cfg.viterbi_lag_s && status === 'OK' ? 'WARMUP' : status,
      fs_proc: fsProc, patches, gridBpm: this.grid, emission, trackBpm: this.grid[vs.current], bvp, row,
      motion: motion ? { rms: motion.rms, peakBpm: 60 * motion.peakHz, active: !!motion.atten } : undefined,
    };
  }

  private analysePatch(
    id: number, group: number, frames: TraceFrame[], tf: number[], tg: Float64Array,
    Lw: number, fsMeasured: number, fsProc: number, reject: boolean[], att: Float64Array | null,
  ): PatchResult {
    const cfg = this.cfg;
    const res: PatchResult = { id, group, coverage: 0, snrDb: -Infinity, peakBpm: NaN, Pn: null, seg: null, weight: 0 };
    const x: number[][] = [], np: number[] = [];
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      const p = reject[i] ? undefined : f.patches.find((q) => q.id === id);
      if (p && p.valid && f.face_status > 0) { x.push([...p.rgb_mean]); np.push(p.npix); }
      else { x.push([NaN, NaN, NaN]); np.push(0); }
    }
    const y = kalmanDenoise(x, fsMeasured, np, cfg.kalman_cutoff_hz, cfg.kalman_nyquist_frac);
    const edge = 1.5 / Math.max(fsMeasured, 1);
    const ch = [0, 1, 2].map((c) => resampleUniform(tf, y.map((r) => r[c]), tg, cfg.resample_max_gap_s, edge));
    const n = tg.length;
    let finite = 0;
    for (let i = n - Lw; i < n; i++) if (Number.isFinite(ch[0][i])) finite++;
    res.coverage = finite / Lw;
    if (res.coverage < cfg.coverage_min) return res;

    const f = ch.map(fillNaN);
    const rgb: Rgb[] = Array.from({ length: n }, (_, i) => [f[0][i], f[1][i], f[2][i]]);
    let pulse: Float64Array;
    try {
      pulse = cfg.method === 'CHROM'
        ? chrom(rgb, fsProc, cfg.band_hz, cfg.filter_order)
        : pos(rgb, Math.round(cfg.pos_window_s * fsProc));
      const lambda = cfg.detrend_lambda * (fsProc / cfg.detrend_lambda_ref_fs) ** 2;
      pulse = zscore(bandpass(detrendSmoothnessPriors(pulse, lambda), fsProc, cfg.band_hz, cfg.filter_order));
    } catch {
      return res;
    }
    const seg = zscore(pulse.slice(n - Lw));
    const { P, nfft } = powerSpectrum(seg, cfg.nfft);
    if (att && att.length === P.length) for (let k = 0; k < P.length; k++) P[k] *= att[k];
    const df = fsProc / nfft;
    const [k0, k1] = bandBins(cfg.band_hz, df, P.length);
    let e = 0;
    for (let k = k0; k <= k1; k++) e += P[k];
    if (!(e > 0)) return res;
    for (let k = 0; k < P.length; k++) P[k] /= e;
    const { snrDb, f0 } = spectralSnr(P, df, cfg.band_hz, cfg.snr_halfwidth_bpm / 60);
    Object.assign(res, { snrDb, peakBpm: 60 * f0, Pn: P, seg });
    return res;
  }

  /** SNR-weighted BVP after sign alignment with the best patch (README 4.7). */
  private fuseBvp(used: PatchResult[], wSum: number): Float64Array {
    const ref = used[0].seg!;
    const out = new Float64Array(ref.length);
    for (const r of used) {
      const s = r.seg!;
      let c = 0;
      for (let i = 0; i < s.length; i++) c += s[i] * ref[i];
      const sign = c < 0 ? -1 : 1;
      for (let i = 0; i < s.length; i++) out[i] += (sign * r.weight * s[i]) / wSum;
    }
    return zscore(out);
  }
}
