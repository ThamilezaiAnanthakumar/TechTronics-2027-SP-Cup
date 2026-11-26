import { describe, expect, it } from 'vitest';
import { CONFIG } from '../src/config';
import { kalmanDenoise } from '../src/core/kalman';

/** Least-squares amplitude of a sinusoid at f in y (skipping the transient). */
function amplitude(y: number[], fs: number, f: number, skip: number) {
  let c = 0, s = 0, cc = 0, ss = 0, cs = 0;
  for (let n = skip; n < y.length; n++) {
    const a = Math.cos((2 * Math.PI * f * n) / fs), b = Math.sin((2 * Math.PI * f * n) / fs);
    const v = y[n] - 100;
    c += v * a; s += v * b; cc += a * a; ss += b * b; cs += a * b;
  }
  const det = cc * ss - cs * cs;
  const A = (c * ss - s * cs) / det, B = (s * cc - c * cs) / det;
  return Math.hypot(A, B);
}

describe('Kalman denoise -3 dB point', () => {
  for (const fs of [30, 10]) {
    const fc = Math.min(CONFIG.kalman_cutoff_hz, (CONFIG.kalman_nyquist_frac * fs) / 2);
    it(`is at ${fc} Hz at ${fs} fps (constant npix)`, () => {
      const N = fs * 60;
      const x = Array.from({ length: N }, (_, n) => [100 + Math.sin((2 * Math.PI * fc * n) / fs + 0.3)]);
      const y = kalmanDenoise(x, fs, new Array(N).fill(400), CONFIG.kalman_cutoff_hz, CONFIG.kalman_nyquist_frac).map((r) => r[0]);
      const gainDb = 20 * Math.log10(amplitude(y, fs, fc, fs * 5));
      expect(gainDb).toBeGreaterThan(-3.01 - 0.1);
      expect(gainDb).toBeLessThan(-3.01 + 0.1);
      // the pulse band is passed nearly untouched
      const xp = Array.from({ length: N }, (_, n) => [100 + Math.sin((2 * Math.PI * 1.2 * n) / fs)]);
      const yp = kalmanDenoise(xp, fs, null, CONFIG.kalman_cutoff_hz, CONFIG.kalman_nyquist_frac).map((r) => r[0]);
      expect(20 * Math.log10(amplitude(yp, fs, 1.2, fs * 5))).toBeGreaterThan(-1);
    });
  }
  it('trusts frames with fewer pixels less (R_k ~ median(npix)/npix_k)', () => {
    const x = [[0], [1], [0], [1], [10]];
    const many = kalmanDenoise(x, 30, [100, 100, 100, 100, 100], 4, 0.8)[4][0];
    const few = kalmanDenoise(x, 30, [100, 100, 100, 100, 10], 4, 0.8)[4][0];
    expect(few).toBeLessThan(many);
  });
  it('keeps NaN rows NaN', () => {
    const y = kalmanDenoise([[1], [NaN], [2], [1.5]], 30, null, 4, 0.8);
    expect(Number.isNaN(y[1][0])).toBe(true);
  });
});
