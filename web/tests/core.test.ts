import { describe, expect, it } from 'vitest';
import { PATCHES } from '../src/config';
import { InterimCore } from '../src/core/interimCore';
import type { HrRow, TraceFrame } from '../src/core/types';
import { mulberry32, syntheticRgb } from './synth';

const T0 = 1000; // arbitrary clock offset: the core must not assume t starts at 0

function frames(bpm: number, seconds: number, fps: number, faceGap?: [number, number]): TraceFrame[] {
  const rand = mulberry32(5);
  const t: number[] = [];
  for (let s = 0; s < seconds; s += (1 / fps) * (0.85 + 0.3 * rand())) t.push(s); // jittered frame times
  const perPatch = PATCHES.map((p) => syntheticRgb(t, bpm, { seed: 100 + p.id, noise: 0.25 }));
  return t.map((rel, i) => {
    const face = !(faceGap && rel >= faceGap[0] && rel < faceGap[1]);
    return {
      t_s: T0 + rel, face_status: face ? 1 : 0, track_quality: face ? 1 : 0, motion: [0, 0, 0, 0],
      n_patches: face ? PATCHES.length : 0,
      patches: face
        ? PATCHES.map((p) => ({
            id: p.id, group: p.groupIndex, cluster: 0, rgb_mean: perPatch[p.id][i], npix: 400, valid: 1, skin_ratio: 0.8,
          }))
        : [],
    };
  });
}

function run(fr: TraceFrame[]) {
  const core = new InterimCore();
  const rows: HrRow[] = [];
  for (const f of fr) for (const tk of core.push(f)) if (tk.row) rows.push(tk.row);
  rows.push(...core.finish());
  return rows;
}

describe('InterimCore end to end on synthetic TraceFrames', () => {
  for (const fps of [30, 10]) {
    it(`outputs 72 +- 1 bpm at 1 Hz after warm-up (${fps} fps, jittered timestamps)`, () => {
      const rows = run(frames(72, 40, fps));
      expect(rows.length).toBeGreaterThanOrEqual(38);
      expect(rows.filter((r) => r.status === 'WARMUP').length).toBeGreaterThanOrEqual(8);
      const ok = rows.filter((r) => r.status === 'OK');
      expect(ok.length).toBeGreaterThanOrEqual(25);
      for (const r of ok) {
        expect(Math.abs(r.hr_bpm - 72)).toBeLessThanOrEqual(1);
        expect(r.confidence).toBeGreaterThan(0.3);
      }
      for (let i = 1; i < rows.length; i++) expect(rows[i].time_s - rows[i - 1].time_s).toBeCloseTo(1, 6);
    });
  }

  it('reports NO_FACE (never silent) while the face is missing', () => {
    const rows = run(frames(72, 40, 30, [20, 26]));
    const gap = rows.filter((r) => r.time_s - T0 > 21 && r.time_s - T0 < 26);
    expect(gap.length).toBeGreaterThan(0);
    for (const r of gap) {
      expect(r.status).toBe('NO_FACE');
      expect(Number.isNaN(r.hr_bpm)).toBe(true);
    }
  });
});

describe('MASKED status (README 4.3.5)', () => {
  const lower = new Set(['P', 'LSM', 'RSM', 'LM', 'RM']);
  function withLower(valid: number, skinRatio: number) {
    return frames(72, 25, 30).map((f) => ({
      ...f,
      patches: f.patches.map((p) => {
        const code = PATCHES.find((q) => q.id === p.id)!.code;
        return lower.has(code) && code !== 'LM' && code !== 'RM' ? { ...p, valid, skin_ratio: skinRatio, rgb_mean: [NaN, NaN, NaN] as [number, number, number] } : p;
      }),
    }));
  }
  it('is set when the lower-face patches are occluded (low skin ratio)', () => {
    const rows = run(withLower(0, 0.1)).filter((r) => r.status !== 'WARMUP');
    expect(rows.length).toBeGreaterThan(5);
    for (const r of rows) expect(r.status).toBe('MASKED');
  });
  it('is not set when they are invalid only because they are small (skin ratio normal)', () => {
    const rows = run(withLower(0, 0.8)).filter((r) => r.status !== 'WARMUP');
    expect(rows.length).toBeGreaterThan(5);
    for (const r of rows) expect(r.status).toBe('OK');
  });
});
