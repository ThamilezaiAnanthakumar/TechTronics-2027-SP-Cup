// TROIKA-style tracker (src/core/troikaTrack.ts): port of tests/test_troika_track.m
// (main repo) and parity with MATLAB src/track/troikaTrack.m on synthetic spectra
// generated identically in both (MATLAB rng(seed) + rand == tests/mt19937.ts).
// Fixture: tests/fixtures/troika_parity.json from tools/export_troika_parity.m.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { colon } from '../src/core/cp3Core';
import { cubicPredictNext, TroikaTracker, type TroikaAction } from '../src/core/troikaTrack';
import { argmaxHr, maxAbsErr, maxStep, scenario, type Scenario } from './trackScenarios';

function runTracker(S: Scenario) {
  const tr = new TroikaTracker();
  const out = S.P.map((p, i) => tr.update(p, S.f, (i + 1) * S.step));
  return { hr: out.map((o) => o.hr), action: out.map((o) => o.action) };
}

describe('TROIKA tracker (port of test_troika_track.m)', () => {
  it('suppresses 85 / 65 bpm motion bursts and follows the real 55 -> 70 bpm rise', () => {
    const S = scenario('motion');
    const am = argmaxHr(S);
    const { hr, action } = runTracker(S);
    const errA = maxAbsErr(am, S.hrTrue), errT = maxAbsErr(hr, S.hrTrue), stepA = maxStep(am), stepT = maxStep(hr);
    console.log(`argmax: max |err| ${errA.toFixed(2)} bpm, max step ${stepA.toFixed(2)} | TROIKA: max |err| ${errT.toFixed(2)} bpm, max step ${stepT.toFixed(2)} bpm, actions ${[...new Set(action)].sort().join(',')}`);
    expect(errA).toBeGreaterThan(20);
    expect(stepT).toBeLessThan(5);
    expect(errT).toBeLessThan(2.5);
    expect(hr.every(Number.isFinite)).toBe(true);
  });

  it('derives the per-window limits from the actual output step (2 s step: clamp at 10 bpm)', () => {
    const S = scenario('step2');
    const { hr } = runTracker(S);
    expect(maxStep(hr)).toBeLessThan(10);
    expect(maxAbsErr(hr, S.hrTrue)).toBeLessThan(2.5);
  });

  it('clamps a jump of >= 5 bpm to 2 bpm, holds without peaks, re-initialises after a > 10 s gap', () => {
    const f = colon(0.7, 0.005, 3.0);
    const peak = (bpm: number) => Float64Array.from(f, (v) => Math.exp(-0.5 * ((60 * v - bpm) / 1.5) ** 2));
    const tr = new TroikaTracker();
    expect(tr.update(peak(60), f, 0).action).toBe('init');
    const c = tr.update(peak(70), f, 1);
    expect(c.action).toBe('clamp');
    expect(Math.abs(c.hr - 62)).toBeLessThan(0.5);
    const h = tr.update(new Float64Array(f.length).fill(1), f, 2); // flat: no local peak
    expect(h).toEqual({ hr: c.hr, action: 'hold' });
    for (let t = 3; t <= 13; t++) expect(tr.update(null, f, t)).toEqual({ hr: NaN, action: 'gap' });
    const r = tr.update(peak(100), f, 14); // 11 s after the last valid window (t = 2): re-initialised at t = 13
    expect(r.action).toBe('init');
    expect(Math.abs(r.hr - 100)).toBeLessThan(0.5);
    tr.reset();
    expect(tr.update(peak(80), f, 20).action).toBe('init');
  });

  it('cubic trend prediction matches polyval(polyfit(x, h, 3), n + 1)', () => {
    expect(cubicPredictNext([1, 8, 27, 64])).toBeCloseTo(125, 8);
    expect(cubicPredictNext([3, 5, 7, 9, 11, 13])).toBeCloseTo(15, 8);
  });
});

interface ParityFixture { scenarios: { name: string; step: number; hr: (number | null)[]; action: string[] }[] }

describe('TROIKA parity with MATLAB troikaTrack.m (identical synthetic input)', () => {
  const J: ParityFixture = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'troika_parity.json'), 'utf8'));
  for (const M of J.scenarios) {
    it(M.name, () => {
      const S = scenario(M.name as 'motion' | 'hard' | 'step2');
      const { hr, action } = runTracker(S);
      expect(hr.length).toBe(M.hr.length);
      let maxDiff = 0, nAct = 0;
      for (let i = 0; i < hr.length; i++) {
        const m = M.hr[i] === null ? NaN : (M.hr[i] as number);
        expect(Number.isFinite(hr[i])).toBe(Number.isFinite(m));
        if (Number.isFinite(m)) maxDiff = Math.max(maxDiff, Math.abs(hr[i] - m));
        if (action[i] === (M.action[i] as TroikaAction)) nAct++;
      }
      console.log(`${M.name}: ${hr.length} windows, max |TS - MATLAB| ${maxDiff} bpm, actions identical ${nAct}/${hr.length}`);
      expect(maxDiff).toBeLessThan(1e-9);
      expect(nAct).toBe(hr.length);
    });
  }
});
