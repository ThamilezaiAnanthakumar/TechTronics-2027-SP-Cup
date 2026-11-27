// CP3 parity with MATLAB src/pipeline/snrFusionHR.m on UBFC DEV subjects
// (fixtures from tools/export_cp3_parity.m in the main repo: the first 60 s of
// cached anatomical-patch traces, per-frame mean RGB only, and MATLAB's window
// centres + HR computed on exactly those rounded numbers).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { butterBandpassBA, cp3Analyse, CP3 } from '../src/core/cp3Core';

interface Fixture {
  subject: string;
  t: number[];
  r: (number | null)[][];
  g: (number | null)[][];
  b: (number | null)[][];
  npix: number[][];
  valid: number[][];
  tc: number[];
  hr: (number | null)[];
  butter_b: number[];
  butter_a: number[];
}

const load = (s: number): Fixture =>
  JSON.parse(readFileSync(join(__dirname, 'fixtures', `cp3_parity_subject${s}.json`), 'utf8'));
const num = (v: number | null) => (v === null ? NaN : v);

function runTs(J: Fixture) {
  const nP = J.r[0].length;
  const X: number[][][] = [], np: number[][] = [];
  for (let q = 0; q < nP; q++) {
    X.push(J.t.map((_, i) => (J.valid[i][q] ? [num(J.r[i][q]), num(J.g[i][q]), num(J.b[i][q])] : [NaN, NaN, NaN])));
    np.push(J.npix.map((row) => row[q]));
  }
  return cp3Analyse(J.t, X, np);
}

describe('CP3 Butterworth coefficients', () => {
  it('match MATLAB butter(3, [0.7 3]/15, "bandpass")', () => {
    const J = load(1);
    const { b, a } = butterBandpassBA(CP3.order, CP3.band, CP3.fs);
    expect(b.length).toBe(J.butter_b.length);
    for (let i = 0; i < b.length; i++) expect(Math.abs(b[i] - J.butter_b[i])).toBeLessThan(1e-12);
    for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - J.butter_a[i]) / Math.max(1, Math.abs(J.butter_a[i]))).toBeLessThan(1e-12);
  });
});

describe('CP3 window HR parity with MATLAB snrFusionHR (UBFC DEV, first 60 s)', () => {
  for (const s of [1, 3, 12]) {
    it(`subject${s}`, () => {
      const J = load(s);
      const res = runTs(J);
      expect(res.tc.length).toBe(J.tc.length);
      for (let i = 0; i < J.tc.length; i++) expect(res.tc[i]).toBeCloseTo(J.tc[i], 9);
      const d: number[] = [];
      for (let i = 0; i < J.hr.length; i++) {
        const m = num(J.hr[i]), w = res.hr[i];
        expect(Number.isFinite(m)).toBe(Number.isFinite(w));
        if (Number.isFinite(m)) d.push(Math.abs(m - w));
      }
      const sorted = [...d].sort((x, y) => x - y);
      const med = sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : 0.5 * (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]);
      const within = d.filter((x) => x <= 0.5).length / d.length;
      console.log(`[cp3 parity] subject${s}: ${d.length} windows, max |diff| ${Math.max(...d).toPrecision(3)} bpm, `
        + `median ${med.toPrecision(3)} bpm, ${(100 * within).toFixed(1)} % within 0.5 bpm`);
      expect(within).toBeGreaterThanOrEqual(0.95);
      expect(med).toBeLessThanOrEqual(0.25);
    });
  }
});
