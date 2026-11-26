// The TS core is a port of deploy/web/rppg.js; check it sample by sample.
// MATLAB golden-vector parity is pending (shared/golden does not exist yet).
import { describe, expect, it } from 'vitest';
import * as seed from '../../deploy/web/rppg.js';
import { bandpass, chrom, detrendSmoothnessPriors, pos } from '../src/core/dsp';
import { syntheticRgb } from './synth';

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

describe('parity with the seed deploy/web/rppg.js', () => {
  const fs = 30;
  const t = Float64Array.from({ length: 15 * fs }, (_, i) => i / fs);
  const rgb = syntheticRgb(t, 84, { seed: 11 });
  const cfg = seed.DEFAULT_CFG;
  it('POS', () => expect(maxAbs(pos(rgb, Math.round(cfg.posWindowSec * fs)), seed.rgbToPulse(rgb, fs, 'POS'))).toBeLessThan(1e-12));
  it('CHROM', () =>
    expect(maxAbs(chrom(rgb, fs, cfg.hrBand, cfg.filterOrder), seed.rgbToPulse(rgb, fs, 'CHROM'))).toBeLessThan(1e-12));
  it('detrend + band-pass', () => {
    const x = rgb.map((v) => v[1]);
    expect(maxAbs(detrendSmoothnessPriors(x, 100), seed.detrendSmoothnessPriors(x, 100))).toBeLessThan(1e-12);
    expect(maxAbs(bandpass(x, fs, cfg.hrBand, cfg.filterOrder), seed.bandpassFilter(x, fs))).toBeLessThan(1e-12);
  });
});
