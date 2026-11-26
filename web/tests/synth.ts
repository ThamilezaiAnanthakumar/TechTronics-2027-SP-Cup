// Synthetic signals only (no real faces or recordings).
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number) {
  return () => {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

/**
 * Skin-like RGB with a pulse along the standard skin pulsatility direction
 * (de Haan & Jeanne 2013), a slow multiplicative illumination drift and
 * independent sensor noise (0..255 units).
 */
export function syntheticRgb(t: ArrayLike<number>, bpm: number, opts: { amp?: number; noise?: number; seed?: number } = {}) {
  const { amp = 0.006, noise = 0.1, seed = 1 } = opts;
  const g = gaussian(mulberry32(seed));
  const dc = [170, 120, 95], pv = [0.33, 0.77, 0.53];
  const out: [number, number, number][] = [];
  for (let i = 0; i < t.length; i++) {
    const s = Math.sin(2 * Math.PI * (bpm / 60) * t[i]) + 0.3 * Math.sin(2 * Math.PI * ((2 * bpm) / 60) * t[i] + 1);
    const drift = 1 + 0.03 * Math.sin(2 * Math.PI * 0.05 * t[i]);
    out.push([0, 1, 2].map((c) => dc[c] * drift * (1 + amp * pv[c] * s) + noise * g()) as [number, number, number]);
  }
  return out;
}

const WARM = [1.25, 1.0, 0.7], COOL = [1.0, 1.3, 0.8];

export interface SwayOpts {
  bpm?: number;
  /** head sway frequency, Hz */
  fm?: number;
  seconds?: number;
  fps?: number;
  /** sway amplitude in IOD (horizontal); vertical is 0.3x, roll 0.02 rad */
  swayIod?: number;
  /** motion-induced additive (specular-like, equal in R, G, B) modulation, 0..255 units at full sway */
  add?: number;
  /** motion-induced multiplicative (shading) modulation at full sway, fraction */
  mul?: number;
  /** quadratic (2 f_m) share of the induced modulation */
  quad?: number;
  /** report the motion in the TraceFrames (false = a front end without motion signals) */
  reportMotion?: boolean;
  seed?: number;
  t0?: number;
}

/**
 * Synthetic TraceFrames of a face swaying periodically in a lighting gradient
 * (no real faces). Every patch carries the pulse plus an intensity modulation
 * driven by the head position with a patch-dependent sign and gain, as a
 * light gradient across the face would produce; the per-frame motion deltas
 * (dx, dy, dscale, droll) are written into the frames.
 */
export function swayFrames(patches: { id: number; groupIndex: number }[], o: SwayOpts = {}) {
  const { bpm = 72, fm = 1.6, seconds = 40, fps = 30, swayIod = 0.05, add = 1.5, mul = 0.01, quad = 0.3,
    reportMotion = true, seed = 7, t0 = 1000 } = o;
  const rand = mulberry32(seed);
  const t: number[] = [];
  for (let s = 0; s < seconds; s += (1 / fps) * (0.9 + 0.2 * rand())) t.push(s);
  const perPatch = patches.map((p) => syntheticRgb(t, bpm, { seed: 100 + p.id, noise: 0.25 }));
  const pose = (s: number) => {
    const ph = 2 * Math.PI * fm * s;
    return [swayIod * Math.sin(ph), 0.3 * swayIod * Math.sin(ph + 0.7), 0.1 * swayIod * Math.sin(ph + 1.1), 0.4 * swayIod * Math.sin(ph + 0.3)];
  };
  let prev = pose(t[0]);
  return t.map((rel, i) => {
    const q = pose(rel);
    const motion = (reportMotion ? q.map((v, c) => v - prev[c]) : [0, 0, 0, 0]) as [number, number, number, number];
    prev = q;
    const u = q[0] / Math.max(swayIod, 1e-12); // -1..1
    const v = q[1] / Math.max(0.3 * swayIod, 1e-12); // -1..1, phase-shifted
    const m = (1 - quad) * u + quad * (u * u - 0.5);
    const mv = (1 - quad) * v + quad * (v * v - 0.5);
    return {
      t_s: t0 + rel, face_status: 1, track_quality: 1, motion, n_patches: patches.length,
      patches: patches.map((p, k) => {
        const gp = Math.cos(1.7 * p.id + 0.4), ga = 0.5 + 0.5 * Math.sin(2.3 * p.id);
        // two differently coloured lights (warm lamp, cool screen): moving in
        // their gradients changes each channel differently, so POS cannot
        // project the modulation out completely.
        const rgb = perPatch[k][i].map((x, c) =>
          x * (1 + mul * (gp * WARM[c] * m + ga * COOL[c] * mv)) + add * ga * m) as [number, number, number];
        return { id: p.id, group: p.groupIndex, cluster: 0, rgb_mean: rgb, npix: 400, valid: 1, skin_ratio: 0.8 };
      }),
    };
  });
}
