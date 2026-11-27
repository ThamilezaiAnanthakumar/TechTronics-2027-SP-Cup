// TROIKA-style tracker (src/core/troikaTrack.ts): port of tests/test_troika_track.m
// (main repo) and parity with MATLAB src/track/troikaTrack.m on synthetic spectra
// generated identically in both (MATLAB rng(seed) + rand == tests/mt19937.ts).
// Fixture: tests/fixtures/troika_parity.json from tools/export_troika_parity.m.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { colon } from '../src/core/cp3Core';
import { cubicPredictNext, TroikaTracker, type TroikaAction } from '../src/core/troikaTrack';
import { matlabRand } from './mt19937';

interface Scenario { P: (Float64Array | null)[]; f: Float64Array; hrTrue: number[]; step: number }

const gauss = (f: Float64Array, c: number) =>
  Float64Array.from(f, (v) => { const z = (60 * v - c) / 1.5; return Math.exp(-0.5 * (z * z)); });

/** Same construction as export_troika_parity.m / test_troika_track.m. */
function scenario(name: 'motion' | 'hard' | 'step2'): Scenario {
  if (name === 'hard') {
    const f: number[] = [];
    for (let k = Math.ceil((0.7 * 8192) / 30); k <= Math.floor((3.0 * 8192) / 30); k++) f.push((k * 30) / 8192);
    const fb = Float64Array.from(f), nW = 200;
    const hrTrue = [...Array(30).fill(70), ...colon(70, 1.2, 100), ...Array(30).fill(100), ...Array.from({ length: 15 }, (_, k) => 100 - 2.5 * k)];
    while (hrTrue.length < nW) hrTrue.push(65);
    const rand = matlabRand(7);
    const P: (Float64Array | null)[] = [];
    for (let w = 1; w <= nW; w++) {
      const a = rand(), b = rand();
      const p = Float64Array.from(fb, () => 0.3 * rand());
      if (w < 120 || w > 127) {
        const g1 = gauss(fb, hrTrue[w - 1]);
        for (let k = 0; k < p.length; k++) p[k] += g1[k];
        if (w % 5 !== 0) { const g2 = gauss(fb, 2 * hrTrue[w - 1]); for (let k = 0; k < p.length; k++) p[k] += 0.4 * g2[k]; }
      }
      if (a < 0.2) { const gm = gauss(fb, 60 + 70 * b); for (let k = 0; k < p.length; k++) p[k] += 2 * gm[k]; }
      const gap = (w >= 60 && w <= 63) || (w >= 150 && w <= 161);
      P.push(gap ? null : p);
    }
    return { P, f: fb, hrTrue, step: 1 };
  }
  const f = colon(0.7, 0.0037, 3.0), nW = 120;
  const hrTrue = [...Array(40).fill(55), ...colon(55, 0.5, 70)];
  while (hrTrue.length < nW) hrTrue.push(70);
  const rand = matlabRand(5);
  const g85 = gauss(f, 85), g65 = gauss(f, 65);
  let P: (Float64Array | null)[] = [];
  for (let w = 1; w <= nW; w++) {
    const g1 = gauss(f, hrTrue[w - 1]), g2 = gauss(f, 2 * hrTrue[w - 1]);
    const p = Float64Array.from(f, (_, k) => g1[k] + 0.4 * g2[k] + 0.05 * rand());
    if (w % 7 === 0) for (let k = 0; k < p.length; k++) p[k] += 1.5 * g85[k];
    if (w % 11 === 0) for (let k = 0; k < p.length; k++) p[k] += 1.5 * g65[k];
    P.push(p);
  }
  let hr = hrTrue, step = 1;
  if (name === 'step2') { P = P.filter((_, i) => i % 2 === 0); hr = hrTrue.filter((_, i) => i % 2 === 0); step = 2; }
  return { P, f, hrTrue: hr, step };
}

function runTracker(S: Scenario) {
  const tr = new TroikaTracker();
  const out = S.P.map((p, i) => tr.update(p, S.f, (i + 1) * S.step));
  return { hr: out.map((o) => o.hr), action: out.map((o) => o.action) };
}
const argmaxHr = (S: Scenario) => S.P.map((p) => {
  if (!p) return NaN;
  let k = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
  return 60 * S.f[k];
});
const maxAbsErr = (a: number[], b: number[]) => Math.max(...a.map((v, i) => (Number.isFinite(v) ? Math.abs(v - b[i]) : 0)));
const maxStep = (a: number[]) => Math.max(...a.slice(1).map((v, i) => (Number.isFinite(v) && Number.isFinite(a[i]) ? Math.abs(v - a[i]) : 0)));

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
