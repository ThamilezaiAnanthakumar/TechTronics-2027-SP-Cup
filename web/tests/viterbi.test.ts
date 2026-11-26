import { describe, expect, it } from 'vitest';
import { CONFIG } from '../src/config';
import { FixedLagViterbi, harmonicSum, hrGrid, sampleOnGrid } from '../src/core/tracker';
import { gaussian, mulberry32 } from './synth';

describe('fixed-lag Viterbi on a synthetic spectrogram with 2x harmonic spikes', () => {
  const fs = 30, nfft = 16384, df = fs / nfft, nBins = nfft / 2 + 1;
  const T = 90;
  const truth = Array.from({ length: T }, (_, t) => 75 + 8 * Math.sin((2 * Math.PI * t) / 60));
  const g = gaussian(mulberry32(7));
  const spikes = new Set<number>();
  for (let t = 12; t < T; t += 6) spikes.add(t);
  const bump = (f: number, f0: number, a: number) => a * Math.exp(-0.5 * ((f - f0) / 0.04) ** 2);
  const spec = truth.map((bpm, t) => {
    const f0 = bpm / 60;
    const P = new Float64Array(nBins);
    const h2 = spikes.has(t) ? 3.0 : 0.4; // injected harmonic spike: 3x the fundamental
    for (let k = 0; k < nBins; k++) {
      const f = k * df;
      P[k] = bump(f, f0, 1) + bump(f, 2 * f0, h2) + 0.02 * Math.abs(g());
    }
    return P;
  });
  const grid = hrGrid(CONFIG);

  it('raw spectral argmax is fooled by every spike (sanity check of the test)', () => {
    let wrong = 0;
    for (const t of spikes) {
      const Pg = sampleOnGrid(spec[t], df, grid);
      let k = 0;
      for (let i = 1; i < Pg.length; i++) if (Pg[i] > Pg[k]) k = i;
      if (Math.abs(grid[k] - truth[t]) > 5) wrong++;
    }
    expect(wrong).toBe(spikes.size);
  });

  it('returns the true track within +-1 bpm at every step', () => {
    const lag = Math.round(CONFIG.viterbi_lag_s / CONFIG.step_s);
    const v = new FixedLagViterbi(grid, CONFIG, lag);
    const decided: number[] = new Array(T).fill(NaN);
    for (let t = 0; t < T; t++) {
      const H = harmonicSum(spec[t], df, CONFIG.harmonic_alpha, (CONFIG.harmonic_nyquist_frac * fs) / 2);
      const d = v.step(sampleOnGrid(H, df, grid)).decided;
      if (d) decided[d.step] = grid[d.state];
    }
    for (const d of v.flush()) decided[d.step] = grid[d.state];
    const err = decided.map((h, t) => Math.abs(h - truth[t]));
    expect(Math.max(...err)).toBeLessThanOrEqual(1);
  });

  it('decoded path respects the -3/+6 bpm/s limits when the evidence jumps', () => {
    const steps = 30;
    const v = new FixedLagViterbi(grid, CONFIG, steps); // lag >= length: flush returns the full path
    const peak = (bpm: number) => Float64Array.from(grid, (h) => Math.exp(-0.5 * ((h - bpm) / 1.5) ** 2) + 1e-3);
    for (let t = 0; t < steps; t++) v.step(peak(t < 10 ? 70 : 120));
    const path = v.flush().map((d) => grid[d.state]);
    expect(path).toHaveLength(steps);
    for (let t = 1; t < steps; t++) {
      expect(path[t] - path[t - 1]).toBeLessThanOrEqual(6 + 1e-9);
      expect(path[t] - path[t - 1]).toBeGreaterThanOrEqual(-3 - 1e-9);
    }
    expect(Math.abs(path[29] - 120)).toBeLessThanOrEqual(1); // it ramps up to the new evidence
    expect(path[11] - path[9]).toBeLessThanOrEqual(12 + 1e-9); // at most +6 bpm/s
  });
});
