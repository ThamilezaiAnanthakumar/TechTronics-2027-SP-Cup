// Stability under small head motion (synthetic face only, no real faces):
// rendered face -> 1-Euro landmark smoothing (as in pipeline.ts) -> FrontEnd
// -> InterimCore. Measures the fraction of 1 Hz outputs after warm-up with
// status OK and |HR - 72| < 5 bpm, still vs +-1-2 px landmark jitter
// (temporally correlated) plus slow drift of the head (image and landmarks).
import { describe, expect, it } from 'vitest';
import { InterimCore } from '../src/core/interimCore';
import type { HrRow } from '../src/core/types';
import { FrontEnd } from '../src/front/frontEnd';
import { OneEuroLandmarks } from '../src/front/oneEuro';
import { faceSequence, type FaceSeqOpts } from './faceSynth';

export function stability(o: FaceSeqOpts) {
  const fe = new FrontEnd(), core = new InterimCore(), euro = new OneEuroLandmarks(1, 0.02, 1);
  const rows: HrRow[] = [];
  for (const fr of faceSequence(o)) {
    const lm = euro.filter(fr.lm, fr.W, fr.H, fr.t);
    const { frame } = fe.process(fr.img, fr.ox, fr.oy, fr.W, fr.H, lm, fr.t, 1);
    for (const tk of core.push(frame)) if (tk.row) rows.push(tk.row);
  }
  rows.push(...core.finish());
  const post = rows.filter((r) => r.status !== 'WARMUP');
  const good = post.filter((r) => r.status === 'OK' && Math.abs(r.hr_bpm - 72) < 5).length;
  return { good, n: post.length, frac: good / Math.max(1, post.length) };
}

describe('stability under small motion (synthetic face through the FrontEnd)', () => {
  const base: FaceSeqOpts = { seconds: 40, fps: 15, pulseAmp: 0.0015, noiseSd: 3, iod: 70, seed: 12 };
  it('keeps >= 80 % of outputs OK and within 5 bpm with +-2 px correlated jitter and slow drift', () => {
    const still = stability(base);
    const moving = stability({
      ...base, jitter: { common: 1.4, indep: 2, ar: 0.8 }, drift: { px: 8, hz: 0.05, rollRad: 0.03 },
    });
    console.log(`stability: still ${still.good}/${still.n}, jitter+drift ${moving.good}/${moving.n}`);
    expect(still.frac).toBeGreaterThanOrEqual(0.8);
    expect(moving.frac).toBeGreaterThanOrEqual(0.8);
  }, 120000);
});
