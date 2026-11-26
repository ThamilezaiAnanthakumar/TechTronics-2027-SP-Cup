import { describe, expect, it } from 'vitest';
import { OneEuroLandmarks } from '../src/front/oneEuro';

// 1 landmark (x, y) in normalised coords on a 640x480 frame, 30 fps
const W = 640, H = 480, fs = 30;
const lm = (xPx: number, yPx: number) => Float32Array.from([xPx / W, yPx / H]);

describe('OneEuroLandmarks', () => {
  it('passes a still point unchanged', () => {
    const f = new OneEuroLandmarks();
    let out = lm(320, 240);
    for (let k = 0; k < 60; k++) out = f.filter(lm(320, 240), W, H, k / fs);
    expect(out[0] * W).toBeCloseTo(320, 6);
    expect(out[1] * H).toBeCloseTo(240, 6);
  });

  it('reduces +-1 px frame-to-frame jitter by at least 4x', () => {
    const f = new OneEuroLandmarks();
    const outs: number[] = [];
    for (let k = 0; k < 300; k++) {
      const jitter = k % 2 === 0 ? 1 : -1;
      outs.push(f.filter(lm(320 + jitter, 240), W, H, k / fs)[0] * W);
    }
    const tail = outs.slice(100);
    const sd = Math.sqrt(tail.reduce((s, v) => s + (v - 320) ** 2, 0) / tail.length);
    expect(sd).toBeLessThan(0.25);
  });

  it('follows a 40 px head movement within 0.5 s', () => {
    const f = new OneEuroLandmarks();
    let x = 0;
    for (let k = 0; k < 30; k++) x = f.filter(lm(320, 240), W, H, k / fs)[0] * W;
    for (let k = 30; k < 45; k++) x = f.filter(lm(360, 240), W, H, k / fs)[0] * W;
    expect(Math.abs(x - 360)).toBeLessThan(4);
  });
});
