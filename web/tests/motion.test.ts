// README 4.8.2 motion suppression on synthetic TraceFrames (no real faces):
// a face sways periodically between two differently coloured lights, so the
// induced intensity modulation survives POS and, without suppression, the
// tracker locks onto the sway (or its 2nd harmonic) instead of the 72 bpm pulse.
import { describe, expect, it } from 'vitest';
import { CONFIG, PATCHES } from '../src/config';
import { InterimCore } from '../src/core/interimCore';
import { motionSpectrum } from '../src/core/motion';
import type { HrRow, TraceFrame } from '../src/core/types';
import { LOCAL_CONFIG, type LocalConfig } from '../src/localConfig';
import { gaussian, mulberry32, swayFrames } from './synth';

const OFF: LocalConfig = { ...LOCAL_CONFIG, motion_suppression: false };

function run(fr: TraceFrame[], local: LocalConfig) {
  const core = new InterimCore(CONFIG, local);
  const rows: HrRow[] = [];
  let active = 0;
  for (const f of fr) for (const tk of core.push(f)) { if (tk.row) rows.push(tk.row); if (tk.motion?.active) active++; }
  rows.push(...core.finish());
  const ok = rows.filter((r) => r.status === 'OK');
  const good = ok.filter((r) => Math.abs(r.hr_bpm - 72) < 5);
  const hr = ok.map((r) => r.hr_bpm).sort((a, b) => a - b);
  return { ok: ok.length, good: good.length, median: hr[hr.length >> 1], active };
}

describe('motion spectrum (README 4.8.2)', () => {
  it('finds the sway rate and attenuates to at most 1 - gamma, never to zero', () => {
    const fr = swayFrames(PATCHES, { fm: 1.6, seconds: 14 });
    const fs = 30, n = 13 * fs, Lw = 10 * fs;
    const T = fr[fr.length - 1].t_s;
    const tg = Float64Array.from({ length: n }, (_, i) => T - (n - 1 - i) / fs);
    const ms = motionSpectrum(fr, fr.map((f) => f.t_s), tg, Lw, fs, CONFIG, LOCAL_CONFIG);
    expect(Math.abs(ms.peakHz - 1.6)).toBeLessThan(0.05);
    expect(ms.atten).not.toBeNull();
    const a = ms.atten!;
    const df = fs / ((a.length - 1) * 2);
    let mn = Infinity;
    for (const v of a) mn = Math.min(mn, v);
    expect(mn).toBeCloseTo(1 - LOCAL_CONFIG.motion_gamma, 6);
    expect(a[Math.round(1.6 / df)]).toBeCloseTo(1 - LOCAL_CONFIG.motion_gamma, 6);
    expect(a[Math.round(1.2 / df)]).toBe(1); // 72 bpm untouched
  });
});

describe('InterimCore with periodic head sway (synthetic)', () => {
  it('1.6 Hz sway: captured at 96 bpm without suppression, 72 bpm recovered with it', () => {
    const fr = swayFrames(PATCHES, { fm: 1.6, mul: 0.02, add: 0, quad: 0, seconds: 40 });
    const off = run(fr, OFF), on = run(fr, LOCAL_CONFIG);
    console.log(`1.6 Hz sway: off ${off.good}/${off.ok} within 5 bpm (median ${off.median.toFixed(1)}), on ${on.good}/${on.ok} (median ${on.median.toFixed(1)})`);
    expect(off.good).toBeLessThanOrEqual(5);
    expect(Math.abs(off.median - 96)).toBeLessThan(3);
    expect(on.good).toBeGreaterThanOrEqual(27);
    expect(Math.abs(on.median - 72)).toBeLessThan(1);
  });

  it('0.5 Hz sway: its 2nd harmonic (60 bpm) captures the tracker without suppression, 72 bpm with it', () => {
    const fr = swayFrames(PATCHES, { fm: 0.5, mul: 0.08, add: 0, quad: 0.5, seconds: 40 });
    const off = run(fr, OFF), on = run(fr, LOCAL_CONFIG);
    console.log(`0.5 Hz sway: off ${off.good}/${off.ok} (median ${off.median.toFixed(1)}), on ${on.good}/${on.ok} (median ${on.median.toFixed(1)})`);
    expect(off.good).toBeLessThanOrEqual(5);
    expect(on.good).toBeGreaterThanOrEqual(27);
    expect(Math.abs(on.median - 72)).toBeLessThan(1);
  });

  it('landmark jitter (broadband) does not trigger suppression', () => {
    const fr = swayFrames(PATCHES, { mul: 0, add: 0, swayIod: 0, seconds: 30 });
    const g = gaussian(mulberry32(9));
    let prev = [0, 0, 0, 0];
    for (const f of fr) {
      const q = [0.02 * g(), 0.02 * g(), 0.006 * g(), 0.01 * g()];
      f.motion = q.map((v, c) => v - prev[c]) as TraceFrame['motion'];
      prev = q;
    }
    const on = run(fr, LOCAL_CONFIG), off = run(fr, OFF);
    expect(on.active).toBe(0);
    expect(on).toEqual(off);
  });
});
