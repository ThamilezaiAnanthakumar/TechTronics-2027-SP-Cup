// Appearance gate (README 4.1) on synthetic images only (no real faces).
import { describe, expect, it } from 'vitest';
import { AppearanceGate, bhattacharyya, centralHistogram } from '../src/front/appearance';
import { LOCAL_CONFIG } from '../src/localConfig';
import { gaussian, mulberry32 } from './synth';

const W = 200, H = 200;
const BOX = { x: 50, y: 40, w: 100, h: 120 };

/** A "face": skin colour with texture, two dark eyes and a lip band, on a grey background. */
function face(skin: number[], gain = 1, seed = 1) {
  const g = gaussian(mulberry32(seed));
  const d = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = 4 * (y * W + x);
    const inFace = x >= BOX.x && x < BOX.x + BOX.w && y >= BOX.y && y < BOX.y + BOX.h;
    const eye = (Math.hypot(x - 80, y - 85) < 7 || Math.hypot(x - 120, y - 85) < 7);
    const lip = Math.abs(y - 135) < 4 && Math.abs(x - 100) < 18;
    const c = !inFace ? [90, 95, 100] : eye ? [40, 30, 28] : lip ? [skin[0] * 0.9, skin[1] * 0.6, skin[2] * 0.65] : skin;
    for (let k = 0; k < 3; k++) d[p + k] = c[k] * gain * (1 + 0.03 * g()) + 2 * g();
    d[p + 3] = 255;
  }
  return d;
}

const hist = (d: Uint8ClampedArray) => centralHistogram(d, W, H, BOX);
const thr = LOCAL_CONFIG.appearance_max_bhattacharyya;

describe('appearance gate', () => {
  it('Bhattacharyya distance is 0 for identical and 1 for disjoint histograms', () => {
    const a = Float64Array.from([0.5, 0.5, 0, 0]), b = Float64Array.from([0, 0, 0.5, 0.5]);
    expect(bhattacharyya(a, a)).toBeCloseTo(0, 6);
    expect(bhattacharyya(a, b)).toBeCloseTo(1, 6);
  });

  it('keeps the same face (new noise, 15 % exposure change) and rejects other skin tones', () => {
    const target = [195, 145, 115];
    const gate = new AppearanceGate(thr);
    gate.update(hist(face(target, 1, 1)));
    const same = gate.distance(hist(face(target, 1, 2)));
    const darker = gate.distance(hist(face(target, 0.85, 3)));
    const others = [[120, 80, 60], [235, 200, 180], [180, 150, 140]].map((s, i) => gate.distance(hist(face(s, 1, 10 + i))));
    console.log(`appearance distances: same ${same.toFixed(3)}, exposure -15 % ${darker.toFixed(3)}, others ${others.map((v) => v.toFixed(3)).join(', ')} (threshold ${thr})`);
    expect(gate.accept(hist(face(target, 1, 2)))).toBe(true);
    expect(gate.accept(hist(face(target, 0.85, 3)))).toBe(true);
    for (const v of others) expect(v).toBeGreaterThan(thr);
  });

  it('accepts anything before a model exists, and the model adapts slowly', () => {
    const gate = new AppearanceGate(thr);
    expect(gate.accept(hist(face([120, 80, 60])))).toBe(true);
    gate.update(hist(face([195, 145, 115])));
    const h2 = hist(face([120, 80, 60]));
    const d0 = gate.distance(h2);
    gate.update(h2);
    expect(gate.distance(h2)).toBeLessThan(d0);
    expect(gate.distance(h2)).toBeGreaterThan(0.8 * d0); // one bystander frame cannot take over
  });
});
