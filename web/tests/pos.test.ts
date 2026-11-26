import { describe, expect, it } from 'vitest';
import { CONFIG } from '../src/config';
import { bandpass, chrom, detrendSmoothnessPriors, pos, powerSpectrum, zscore } from '../src/core/dsp';
import { spectralSnr } from '../src/core/tracker';
import { syntheticRgb } from './synth';

function hrOf(pulse: Float64Array, fs: number) {
  const p = zscore(bandpass(detrendSmoothnessPriors(pulse, CONFIG.detrend_lambda), fs, CONFIG.band_hz, CONFIG.filter_order));
  const seg = p.slice(p.length - Math.round(CONFIG.window_s * fs));
  const { P, nfft } = powerSpectrum(seg, CONFIG.nfft);
  return 60 * spectralSnr(P, fs / nfft, CONFIG.band_hz, 0.1).f0;
}

describe('POS on a synthetic 72 bpm RGB pulse with noise', () => {
  const fs = 30;
  const t = Float64Array.from({ length: 20 * fs }, (_, i) => i / fs);
  for (const seed of [1, 2, 3]) {
    it(`recovers 72 +- 0.5 bpm (seed ${seed})`, () => {
      const rgb = syntheticRgb(t, 72, { seed });
      const hr = hrOf(pos(rgb, Math.round(CONFIG.pos_window_s * fs)), fs);
      expect(Math.abs(hr - 72)).toBeLessThanOrEqual(0.5);
    });
  }
  it('CHROM option also recovers 72 +- 0.5 bpm', () => {
    const rgb = syntheticRgb(t, 72, { seed: 4 });
    const hr = hrOf(chrom(rgb, fs, CONFIG.band_hz, CONFIG.filter_order), fs);
    expect(Math.abs(hr - 72)).toBeLessThanOrEqual(0.5);
  });
});
