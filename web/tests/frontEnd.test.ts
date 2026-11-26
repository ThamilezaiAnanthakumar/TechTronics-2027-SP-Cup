// Front end on synthetic frames (no real faces): uniform skin-coloured image
// with a pulse, per-pixel noise, a specular spot and dark "hair" outliers.
import { describe, expect, it } from 'vitest';
import { PATCHES } from '../src/config';
import { InterimCore } from '../src/core/interimCore';
import type { HrRow } from '../src/core/types';
import { ita, srgbToLab } from '../src/front/color';
import { FrontEnd, trimmedMeanRgb } from '../src/front/frontEnd';
import { rasterise } from '../src/front/geometry';
import { gaussian, mulberry32 } from './synth';

const W = 160, H = 120;
const SKIN = [200, 150, 120];

/** 478 synthetic landmarks scattered over the central part of the frame. */
function syntheticLandmarks(): Float32Array {
  const lm = new Float32Array(478 * 2);
  for (let k = 0; k < 478; k++) {
    lm[2 * k] = 0.25 + 0.5 * ((k * 0.6180339) % 1);
    lm[2 * k + 1] = 0.25 + 0.5 * ((k * 0.4142135) % 1);
  }
  return lm;
}

function frame(t: number, g: () => number, bpm = 72) {
  const data = new Uint8ClampedArray(W * H * 4);
  const s = 1 + 0.006 * Math.sin(2 * Math.PI * (bpm / 60) * t);
  const pv = [0.33, 0.77, 0.53];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = 4 * (y * W + x);
    const spec = Math.hypot(x - 80, y - 60) < 4; // specular highlight
    const hair = (x * 7 + y * 13) % 53 === 0;   // sparse dark outliers
    for (let c = 0; c < 3; c++) {
      const v = spec ? 255 : hair ? 30 : SKIN[c] * (1 + (s - 1) * pv[c] / 0.77) + 3 * g();
      data[p + c] = v;
    }
    data[p + 3] = 255;
  }
  return { width: W, height: H, data, colorSpace: 'srgb' } as ImageData;
}

describe('rasterise', () => {
  it('fills an axis-aligned square with exactly its pixels', () => {
    const sq = Float32Array.from([2, 2, 12, 2, 12, 12, 2, 12]);
    expect(rasterise(sq, 0, 0, 20, 20).length).toBe(100);
  });
});

describe('trimmed mean', () => {
  it('drops 10 % at each end per channel', () => {
    const d = new Uint8ClampedArray(10 * 4);
    [0, 100, 100, 100, 100, 100, 100, 100, 100, 255].forEach((v, i) => { d[4 * i] = v; d[4 * i + 1] = v; d[4 * i + 2] = v; });
    expect(trimmedMeanRgb(d, [...Array(10).keys()], 0.1)[0]).toBe(100);
  });
});

describe('FrontEnd -> InterimCore on synthetic frames', () => {
  it('emits valid TraceFrames, fits the skin model, reads ITA and recovers 72 bpm', () => {
    const fe = new FrontEnd();
    const core = new InterimCore();
    const lm = syntheticLandmarks();
    const g = gaussian(mulberry32(3));
    const rows: HrRow[] = [];
    let last: ReturnType<FrontEnd['process']> | null = null;
    const fps = 15;
    for (let i = 0; i < 30 * fps; i++) {
      const t = i / fps;
      last = fe.process(frame(t, g), 0, 0, W, H, lm, t, 1);
      for (const tk of core.push(last.frame)) if (tk.row) rows.push(tk.row);
    }
    rows.push(...core.finish());
    const f = last!.frame;
    expect(f.n_patches).toBe(PATCHES.length);
    expect(last!.info.skinModelFitted).toBe(true);
    for (const p of f.patches) {
      expect(p.valid).toBe(1);
      expect(Math.abs(p.rgb_mean[1] - SKIN[1])).toBeLessThan(3); // specular + outliers rejected
      expect(p.skin_ratio).toBeGreaterThan(0.6);
      expect(p.skin_ratio).toBeLessThan(0.9); // texture exclusion removed ~15 %
    }
    const lab = srgbToLab(SKIN[0], SKIN[1], SKIN[2]);
    expect(Math.abs(last!.info.itaDeg - ita(lab[0], lab[2]))).toBeLessThan(2);
    const ok = rows.filter((r) => r.status === 'OK');
    expect(ok.length).toBeGreaterThan(10);
    for (const r of ok) expect(Math.abs(r.hr_bpm - 72)).toBeLessThanOrEqual(1.5);
  });
});
