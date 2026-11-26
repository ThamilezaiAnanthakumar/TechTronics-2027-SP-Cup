// 2x2 sub-patches as fusion units and README 4.4 adaptive pooling, on
// synthetic TraceFrames (no real faces).
import { describe, expect, it } from 'vitest';
import { CONFIG, PATCHES } from '../src/config';
import { InterimCore } from '../src/core/interimCore';
import type { HrRow, TraceFrame, TracePatch } from '../src/core/types';
import { LOCAL_CONFIG, type LocalConfig } from '../src/localConfig';
import { mulberry32, syntheticRgb } from './synth';

type Rgb = [number, number, number];

/**
 * Frames whose sub-patches carry the pulse; `bad(q)` sub-patches additionally
 * flicker at 1.6 Hz with a colour change (e.g. a strand of hair or a moving
 * shadow over one corner of every patch). The parent patch is the npix-weighted
 * mean of its sub-patches, as the front end would measure it.
 */
function frames(opts: { bad?: (q: number) => boolean; subNpix?: number; flicker?: number; seconds?: number; noise?: number }): TraceFrame[] {
  const { bad = () => false, subNpix = 100, flicker = 0.06, seconds = 40, noise = 0.5 } = opts;
  const rand = mulberry32(3);
  const t: number[] = [];
  for (let s = 0; s < seconds; s += (1 / 30) * (0.9 + 0.2 * rand())) t.push(s);
  const traces = PATCHES.flatMap((p) => [0, 1, 2, 3].map((q) => syntheticRgb(t, 72, { seed: 500 + 4 * p.id + q, noise })));
  const npixMin = CONFIG.npix_min_qvga * 4; // VGA
  return t.map((rel, i) => {
    const subs: TracePatch[] = [];
    const patches: TracePatch[] = PATCHES.map((p) => {
      const parts: { rgb: Rgb; n: number }[] = [];
      for (let q = 0; q < 4; q++) {
        let rgb = traces[4 * p.id + q][i];
        if (bad(q)) {
          // two colour directions in quadrature, so POS cannot project it out
          const ph = 2 * Math.PI * 1.6 * rel;
          rgb = rgb.map((v, c) => v * (1 + flicker * (Math.sin(ph) * [1.25, 1.0, 0.7][c] + Math.cos(ph) * [1.0, 1.3, 0.8][c]))) as Rgb;
        }
        const valid = subNpix >= npixMin ? 1 : 0;
        subs.push({ id: 4 * p.id + q, group: p.groupIndex, cluster: 0, rgb_mean: valid ? rgb : [NaN, NaN, NaN], npix: subNpix, valid, skin_ratio: 0.8 });
        parts.push({ rgb, n: subNpix });
      }
      const n = parts.reduce((s, x) => s + x.n, 0);
      const rgb = [0, 1, 2].map((c) => parts.reduce((s, x) => s + x.rgb[c] * x.n, 0) / n) as Rgb;
      return { id: p.id, group: p.groupIndex, cluster: 0, rgb_mean: rgb, npix: n, valid: n >= npixMin ? 1 : 0, skin_ratio: 0.8 };
    });
    return { t_s: 1000 + rel, face_status: 1, track_quality: 1, motion: [0, 0, 0, 0], n_patches: patches.length, patches, subpatches: subs };
  });
}

function run(fr: TraceFrame[], local: LocalConfig) {
  const core = new InterimCore(CONFIG, local);
  const rows: HrRow[] = [];
  let subUnits = 0, ticks = 0;
  for (const f of fr) for (const tk of core.push(f)) {
    if (tk.row) rows.push(tk.row);
    if (tk.patches.length) { ticks++; subUnits += tk.patches.filter((p) => p.sub).length; }
  }
  rows.push(...core.finish());
  const post = rows.filter((r) => r.status !== 'WARMUP');
  const good = post.filter((r) => r.status === 'OK' && Math.abs(r.hr_bpm - 72) < 5).length;
  return { good, n: post.length, statuses: [...new Set(post.map((r) => r.status))], meanSubUnits: subUnits / Math.max(1, ticks) };
}

const PATCH_LEVEL: LocalConfig = { ...LOCAL_CONFIG, subpatch_fusion: false };

describe('sub-patch fusion units (README 4.2) and adaptive pooling (README 4.4)', () => {
  it('a flickering corner in every patch captures patch-level fusion; sub-patch fusion drops that corner', () => {
    const fr = frames({ bad: (q) => q === 0 });
    const patch = run(fr, PATCH_LEVEL), sub = run(fr, LOCAL_CONFIG);
    console.log(`flickering corner: patch-level ${patch.good}/${patch.n}, sub-patch ${sub.good}/${sub.n} OK within 5 bpm (mean sub-units ${sub.meanSubUnits.toFixed(1)})`);
    expect(patch.good).toBeLessThanOrEqual(patch.n / 2);
    expect(sub.good).toBeGreaterThanOrEqual(sub.n - 2);
  });

  it('sub-patches below npix_min are pooled into their parent instead of losing coverage', () => {
    const fr = frames({ subNpix: 60, seconds: 30 }); // sub-patch invalid (< 80 px at VGA), parent valid (240 px)
    const sub = run(fr, LOCAL_CONFIG);
    expect(sub.meanSubUnits).toBe(0);
    expect(sub.good).toBeGreaterThanOrEqual(sub.n - 1);
  });

  it('clean sub-patches stay split (no pooling without cause)', () => {
    const sub = run(frames({ seconds: 30, noise: 0.25 }), LOCAL_CONFIG);
    expect(sub.statuses).toEqual(['OK']);
    expect(sub.meanSubUnits).toBeGreaterThan(30);
  });
});

