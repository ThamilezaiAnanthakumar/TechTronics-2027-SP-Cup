// motion.ts — README 4.8.2 spectral motion suppression for the interim core.
//
// The TraceFrames carry per-frame motion deltas (dx, dy in IOD, dscale,
// droll in rad). Per window we integrate them to head pose, resample onto the
// processing grid, pass each channel through the same detrend + band-pass as
// the pulse, and take the Hann power spectrum on the same FFT grid as the patch
// spectra. M(f) = summed motion power / its in-band maximum (0..1). The fused
// spectrum is attenuated by (1 - gamma M(f)) where M(f) > beta; gamma < 1 so it
// is never zeroed. Optionally M also covers 2x the motion frequencies, taken
// from a wider (band_lo / 2 .. band_hi) motion spectrum, because sway below the
// pulse band still modulates reflected light at its second harmonic.
import type { FvConfig } from '../config';
import type { LocalConfig } from '../localConfig';
import { bandpass, detrendSmoothnessPriors, powerSpectrum } from './dsp';
import { bandBins, spectralSnr } from './tracker';
import type { TraceFrame } from './types';

export interface MotionSpectrum {
  /** Multiplicative attenuation per FFT bin (1 = untouched); null when gated off. */
  atten: Float64Array | null;
  /** Band-passed RMS head motion over the window (gate quantity). */
  rms: number;
  /** Dominant motion frequency (Hz) in the wide band, NaN if none. */
  peakHz: number;
  /** Periodicity of the motion (spectral SNR of the wide-band motion, dB). */
  snrDb: number;
}

/** Fill NaNs by linear interpolation (hold at the ends). */
export function fillNaN(x: Float64Array): Float64Array {
  const y = Float64Array.from(x);
  let last = -1;
  for (let i = 0; i < y.length; i++) {
    if (!Number.isFinite(y[i])) continue;
    if (last < 0) for (let k = 0; k < i; k++) y[k] = y[i];
    else if (i - last > 1) for (let k = last + 1; k < i; k++) y[k] = y[last] + ((y[i] - y[last]) * (k - last)) / (i - last);
    last = i;
  }
  if (last >= 0) for (let k = last + 1; k < y.length; k++) y[k] = y[last];
  return y;
}

/** Cumulative head pose (4 channels) at each frame, from the per-frame deltas. */
export function integrateMotion(frames: TraceFrame[]): number[][] {
  const acc = [0, 0, 0, 0];
  const out: number[][] = [[], [], [], []];
  for (const f of frames) {
    if (f.face_status > 0) for (let c = 0; c < 4; c++) acc[c] += Number.isFinite(f.motion[c]) ? f.motion[c] : 0;
    for (let c = 0; c < 4; c++) out[c].push(f.face_status > 0 ? acc[c] : NaN);
  }
  return out;
}

/** Linear interpolation of (t, v) onto tg, NaN samples skipped, ends held. */
function interp(t: number[], v: number[], tg: Float64Array): Float64Array {
  const ti: number[] = [], vi: number[] = [];
  for (let i = 0; i < t.length; i++) if (Number.isFinite(v[i])) { ti.push(t[i]); vi.push(v[i]); }
  const out = new Float64Array(tg.length).fill(NaN);
  if (ti.length < 2) return out;
  let j = 0;
  for (let i = 0; i < tg.length; i++) {
    const q = tg[i];
    if (q <= ti[0]) { out[i] = vi[0]; continue; }
    if (q >= ti[ti.length - 1]) { out[i] = vi[vi.length - 1]; continue; }
    while (j < ti.length - 2 && ti[j + 1] < q) j++;
    const w = (q - ti[j]) / (ti[j + 1] - ti[j]);
    out[i] = vi[j] * (1 - w) + vi[j + 1] * w;
  }
  return out;
}

/**
 * Motion attenuation for one analysis window.
 * frames/tf: frames in the window and their timestamps; tg: processing grid
 * (length n, pad included); Lw: analysed samples at the end of tg.
 */
export function motionSpectrum(
  frames: TraceFrame[], tf: number[], tg: Float64Array, Lw: number, fsProc: number,
  cfg: FvConfig, local: LocalConfig,
): MotionSpectrum {
  const n = tg.length;
  const pose = integrateMotion(frames);
  const lambda = cfg.detrend_lambda * (fsProc / cfg.detrend_lambda_ref_fs) ** 2;
  const wide: [number, number] = [cfg.band_hz[0] / 2, cfg.band_hz[1]];
  let Psame: Float64Array | null = null, Pwide: Float64Array | null = null;
  let nfft = 0, varWide = 0;
  for (let c = 0; c < 4; c++) {
    const x = interp(tf, pose[c], tg);
    if (!Number.isFinite(x[0])) continue;
    const d = detrendSmoothnessPriors(fillNaN(x), lambda);
    let same: Float64Array, wb: Float64Array;
    try {
      same = bandpass(d, fsProc, cfg.band_hz, cfg.filter_order).slice(n - Lw);
      wb = bandpass(d, fsProc, wide, cfg.filter_order).slice(n - Lw);
    } catch {
      continue;
    }
    let m = 0;
    for (const v of wb) m += v;
    m /= wb.length;
    for (const v of wb) varWide += (v - m) ** 2 / wb.length;
    const s = powerSpectrum(same, cfg.nfft), w = powerSpectrum(wb, cfg.nfft);
    nfft = s.nfft;
    if (!Psame) { Psame = s.P; Pwide = w.P; }
    else for (let k = 0; k < s.P.length; k++) { Psame[k] += s.P[k]; Pwide![k] += w.P[k]; }
  }
  const rms = Math.sqrt(varWide);
  if (!Psame || !Pwide) return { atten: null, rms: 0, peakHz: NaN, snrDb: -Infinity };
  const df = fsProc / nfft;
  const [k0, k1] = bandBins(cfg.band_hz, df, Psame.length);
  const [w0, w1] = bandBins(wide, df, Pwide.length);
  let maxS = 0, maxW = 0, kw = w0;
  for (let k = k0; k <= k1; k++) maxS = Math.max(maxS, Psame[k]);
  for (let k = w0; k <= w1; k++) if (Pwide[k] > maxW) { maxW = Pwide[k]; kw = k; }
  const peakHz = maxW > 0 ? kw * df : NaN;
  const snrDb = spectralSnr(Pwide, df, wide, cfg.snr_halfwidth_bpm / 60).snrDb;
  if (!local.motion_suppression || rms < local.motion_min_rms || snrDb < local.motion_min_snr_db || !(maxS > 0)) {
    return { atten: null, rms, peakHz, snrDb };
  }

  const Mk = new Float64Array(Psame.length);
  for (let k = k0; k <= k1; k++) {
    let M = Psame[k] / maxS;
    if (local.motion_harmonic && maxW > 0) {
      // second harmonic of the dominant motion: the wide spectrum translated
      // by f_peak (keeps the main-lobe width, unlike stretching f -> f/2)
      const kh = k - kw;
      if (kh >= w0 && kh <= w1) M = Math.max(M, Pwide[kh] / maxW);
    }
    Mk[k] = Math.min(1, M);
  }
  const r = Math.round(local.motion_dilate_bpm / 60 / df);
  const atten = new Float64Array(Psame.length).fill(1);
  for (let k = k0; k <= k1; k++) {
    let M = Mk[k];
    for (let j = Math.max(k0, k - r); j <= Math.min(k1, k + r); j++) M = Math.max(M, Mk[j]);
    if (M > local.motion_beta) atten[k] = 1 - local.motion_gamma * M;
  }
  return { atten, rms, peakHz, snrDb };
}

/** Frames whose inter-frame motion step exceeds the rejection threshold. */
export function largeMotionFrames(frames: TraceFrame[], step: number): boolean[] {
  return frames.map((f) => step > 0 && f.face_status > 0 && (Math.hypot(f.motion[0], f.motion[1]) > step || Math.abs(f.motion[2]) > step));
}
