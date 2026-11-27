// Viterbi HR tracker (src/core/viterbiTrack.ts, port of src/signal/viterbiHR.m with
// MaxDown = MaxUp = 4.99 bpm/s): the default tracker of the CP3 core's 1 Hz output.
// Offline parity with MATLAB on the synthetic scenarios of tools/export_troika_parity.m
// and the online fixed-lag (3 s) tracker on the 55 -> 70 bpm scenario with 85 / 65
// bpm motion bursts.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { viterbiHR, ViterbiTracker } from '../src/core/viterbiTrack';
import { argmaxHr, maxAbsErr, maxStep, scenario, type Scenario } from './trackScenarios';

function runOnline(S: Scenario, lagS = 3) {
  const tr = new ViterbiTracker({ lagS: lagS * S.step, stepS: S.step });
  const hr = new Array<number>(S.P.length).fill(NaN);
  let clamped = 0;
  const take = (ds: ReturnType<ViterbiTracker['update']>) => { for (const d of ds) { hr[d.index] = d.hr; if (d.clamped) clamped++; } };
  S.P.forEach((p, i) => take(tr.update(p, S.f, (i + 1) * S.step)));
  take(tr.flush());
  return { hr, clamped };
}

describe('Viterbi tracker (default, symmetric 4.99 bpm/s, lag 3 s)', () => {
  it('suppresses 85 / 65 bpm motion bursts and follows the real 55 -> 70 bpm rise', () => {
    const S = scenario('motion');
    const am = argmaxHr(S);
    const { hr, clamped } = runOnline(S);
    const errA = maxAbsErr(am, S.hrTrue), errV = maxAbsErr(hr, S.hrTrue), stepV = maxStep(hr);
    console.log(`argmax: max |err| ${errA.toFixed(2)} bpm, max step ${maxStep(am).toFixed(2)} | Viterbi online (lag 3 s): max |err| ${errV.toFixed(2)} bpm, max step ${stepV.toFixed(2)} bpm, ${clamped} clamped outputs`);
    expect(errA).toBeGreaterThan(20);
    expect(hr.every(Number.isFinite)).toBe(true);
    expect(stepV).toBeLessThan(5);
    expect(errV).toBeLessThan(2.5);
  });

  it('online limits scale with the actual step (2 s step: < 10 bpm per reading)', () => {
    const S = scenario('step2');
    const { hr } = runOnline(S);
    expect(maxStep(hr)).toBeLessThan(10);
    expect(maxAbsErr(hr, S.hrTrue)).toBeLessThan(2.5);
  });

  it('online with a lag longer than the record + flush equals the offline full Viterbi', () => {
    const S = scenario('hard');
    const off = viterbiHR(S.P, S.f);
    const tr = new ViterbiTracker({ lagS: 1000, gapResetS: Infinity });
    const hr = new Array<number>(S.P.length).fill(NaN);
    S.P.forEach((p, i) => { for (const d of tr.update(p, S.f, i + 1)) hr[d.index] = d.hr; });
    for (const d of tr.flush()) hr[d.index] = d.hr;
    for (let i = 0; i < hr.length; i++) expect(hr[i]).toBeCloseTo(off.hr[i], 9);
  });

  it('restarts after more than 10 s without a spectrum (flushing the pending decisions)', () => {
    const S = scenario('motion');
    const tr = new ViterbiTracker();
    let n = 0;
    for (let i = 0; i < 20; i++) n += tr.update(S.P[i], S.f, i + 1).length;
    expect(n).toBe(17);
    for (let t = 21; t <= 30; t++) n += tr.update(null, S.f, t).length; // gap steps (flat emission) are decided too
    expect(n).toBe(27);
    const out = tr.update(null, S.f, 31);                              // 11 s since the last spectrum
    expect(out.map((d) => d.index)).toEqual([27, 28, 29]);             // flushed
    expect(tr.stepCount).toBe(0);
    expect(tr.update(S.P[40], S.f, 33)).toEqual([]);                  // new path, inside the lag
  });
});

interface ParityFixture { scenarios: { name: string; step: number; viterbi: number[] }[] }

describe('Viterbi offline parity with MATLAB viterbiHR.m (MaxDown = MaxUp = 4.99)', () => {
  const J: ParityFixture = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'troika_parity.json'), 'utf8'));
  for (const M of J.scenarios) {
    it(M.name, () => {
      const S = scenario(M.name as 'motion' | 'hard' | 'step2');
      const { hr } = viterbiHR(S.P, S.f, { step: S.step });
      let maxDiff = 0;
      for (let i = 0; i < hr.length; i++) maxDiff = Math.max(maxDiff, Math.abs(hr[i] - M.viterbi[i]));
      console.log(`${M.name}: ${hr.length} windows, max |TS - MATLAB| ${maxDiff.toExponential(2)} bpm; offline max |err| ${maxAbsErr(hr, S.hrTrue).toFixed(2)}, max step ${maxStep(hr).toFixed(2)}`);
      expect(maxDiff).toBeLessThan(1e-6);
    });
  }
});
