// Cp3Core (online CP3) on synthetic TraceFrames, and online vs MATLAB offline
// on the UBFC DEV parity fixtures.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { PATCHES } from '../src/config';
import { Cp3Core } from '../src/core/cp3Core';
import type { HrRow, TraceFrame } from '../src/core/types';
import { mulberry32, syntheticRgb } from './synth';

const T0 = 1000;

function frames(bpm: number, seconds: number, fps: number, faceGap?: [number, number]): TraceFrame[] {
  const rand = mulberry32(5);
  const t: number[] = [];
  for (let s = 0; s < seconds; s += (1 / fps) * (0.85 + 0.3 * rand())) t.push(s);
  const perPatch = PATCHES.map((p) => syntheticRgb(t, bpm, { seed: 100 + p.id, noise: 0.25 }));
  return t.map((rel, i) => {
    const face = !(faceGap && rel >= faceGap[0] && rel < faceGap[1]);
    return {
      t_s: T0 + rel, face_status: face ? 1 : 0, track_quality: face ? 1 : 0, motion: [0, 0, 0, 0],
      n_patches: face ? PATCHES.length : 0,
      patches: face
        ? PATCHES.map((p) => ({ id: p.id, group: p.groupIndex, cluster: 0, rgb_mean: perPatch[p.id][i], npix: 400, valid: 1, skin_ratio: 0.8 }))
        : [],
    };
  });
}

function run(fr: TraceFrame[]) {
  const core = new Cp3Core();
  const rows: HrRow[] = [];
  for (const f of fr) for (const tk of core.push(f)) if (tk.row) rows.push(tk.row);
  rows.push(...core.finish());
  return rows;
}

describe('Cp3Core end to end on synthetic TraceFrames', () => {
  for (const fps of [30, 10]) {
    it(`recovers 72 +- 1 bpm at 1 Hz after warm-up (${fps} fps, jittered timestamps)`, () => {
      const rows = run(frames(72, 40, fps));
      expect(rows.length).toBeGreaterThanOrEqual(38);
      const warm = rows.filter((r) => r.status === 'WARMUP');
      expect(warm.length).toBeGreaterThanOrEqual(9);
      const ok = rows.filter((r) => r.status === 'OK');
      expect(ok.length).toBeGreaterThanOrEqual(28);
      const err = ok.map((r) => Math.abs(r.hr_bpm - 72));
      console.log(`[cp3 synth ${fps} fps] ${ok.length} OK rows, max |err| ${Math.max(...err).toFixed(3)} bpm, `
        + `confidence ${Math.min(...ok.map((r) => r.confidence)).toFixed(2)}-${Math.max(...ok.map((r) => r.confidence)).toFixed(2)}`);
      for (const r of ok) {
        expect(Math.abs(r.hr_bpm - 72)).toBeLessThanOrEqual(1);
        expect(r.confidence).toBeGreaterThan(0.3);
        expect(r.confidence).toBeLessThanOrEqual(1);
      }
      for (const r of rows) if (r.status !== 'OK' && r.status !== 'MASKED') expect(Number.isNaN(r.hr_bpm)).toBe(true);
      for (let i = 1; i < rows.length; i++) expect(rows[i].time_s - rows[i - 1].time_s).toBeCloseTo(1, 6);
    });
  }

  it('reports NO_FACE (never OK without an HR) while the face is missing', () => {
    const rows = run(frames(72, 40, 30, [20, 26]));
    const gap = rows.filter((r) => r.time_s - T0 > 21 && r.time_s - T0 < 26);
    expect(gap.length).toBeGreaterThan(0);
    for (const r of gap) {
      expect(r.status).toBe('NO_FACE');
      expect(Number.isNaN(r.hr_bpm)).toBe(true);
    }
    // right after the gap the window lacks samples -> LOW_COVERAGE, not OK
    for (const r of rows) if (r.status === 'OK') expect(Number.isFinite(r.hr_bpm)).toBe(true);
  });
});

describe('Cp3Core online vs MATLAB offline (UBFC DEV fixtures)', () => {
  for (const s of [1, 3, 12]) {
    it(`subject${s}: online rows within the offline windows`, () => {
      const J = JSON.parse(readFileSync(join(__dirname, 'fixtures', `cp3_parity_subject${s}.json`), 'utf8'));
      const n = (v: number | null) => (v === null ? NaN : v);
      const fr: TraceFrame[] = J.t.map((t: number, i: number) => ({
        t_s: t, face_status: 1, track_quality: 1, motion: [0, 0, 0, 0], n_patches: 10,
        patches: PATCHES.map((p, q) => ({
          id: p.id, group: p.groupIndex, cluster: 0, rgb_mean: [n(J.r[i][q]), n(J.g[i][q]), n(J.b[i][q])],
          npix: J.npix[i][q], valid: J.valid[i][q], skin_ratio: 0.8,
        })),
      }));
      const rows = run(fr);
      const d: number[] = [];
      let lowSnr = 0, okN = 0;
      for (const r of rows) {
        if (r.status === 'LOW_SNR') lowSnr++;
        if (r.status !== 'OK') continue;
        okN++;
        const k = J.tc.findIndex((c: number) => Math.abs(c - (r.time_s - 5)) < 0.02);
        if (k >= 0 && J.hr[k] !== null) d.push(Math.abs(r.hr_bpm - J.hr[k]));
      }
      d.sort((a, b) => a - b);
      const med = d[Math.floor(d.length / 2)];
      console.log(`[cp3 online] subject${s}: ${rows.length} rows, ${okN} OK, ${lowSnr} LOW_SNR; vs MATLAB offline `
        + `(${d.length} matched windows): median |diff| ${med.toFixed(2)} bpm, max ${d[d.length - 1].toFixed(2)} bpm, `
        + `${((100 * d.filter((x) => x <= 0.5).length) / d.length).toFixed(0)} % within 0.5 bpm`);
      expect(d.length).toBeGreaterThan(20);
    });
  }
});
