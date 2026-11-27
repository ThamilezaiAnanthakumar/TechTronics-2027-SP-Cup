// Synthetic tracker scenarios shared by troikaTrack.test.ts and viterbiTrack.test.ts,
// built exactly as tools/export_troika_parity.m (main repo) builds them in MATLAB.
import { colon } from '../src/core/cp3Core';
import { matlabRand } from './mt19937';

export interface Scenario { P: (Float64Array | null)[]; f: Float64Array; hrTrue: number[]; step: number }

const gauss = (f: Float64Array, c: number) =>
  Float64Array.from(f, (v) => { const z = (60 * v - c) / 1.5; return Math.exp(-0.5 * (z * z)); });

/** Same construction as export_troika_parity.m / test_troika_track.m. */
export function scenario(name: 'motion' | 'hard' | 'step2'): Scenario {
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

export const maxAbsErr = (a: number[], b: number[]) => Math.max(...a.map((v, i) => (Number.isFinite(v) ? Math.abs(v - b[i]) : 0)));
export const maxStep = (a: number[]) => Math.max(...a.slice(1).map((v, i) => (Number.isFinite(v) && Number.isFinite(a[i]) ? Math.abs(v - a[i]) : 0)));
export const argmaxHr = (S: Scenario) => S.P.map((p) => {
  if (!p) return NaN;
  let k = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
  return 60 * S.f[k];
});
