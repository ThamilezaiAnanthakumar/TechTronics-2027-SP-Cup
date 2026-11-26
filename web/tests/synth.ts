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
