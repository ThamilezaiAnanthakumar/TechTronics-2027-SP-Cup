// Synthetic face renderer (no real faces), free of config imports so the
// Playwright smoke test can use it too. See faceSynth.ts for the landmarks.
const SKIN = [195, 145, 115], BG = [60, 72, 95], DARK = [45, 35, 32], BROW = [80, 58, 45], LIP = [160, 80, 78];
const PV = [0.33, 0.77, 0.53];

function hash2(i: number, j: number): number {
  let h = Math.imul(i, 374761393) ^ Math.imul(j, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296 - 0.5;
}

const smooth = (e: number) => (e <= 0 ? 0 : e >= 1 ? 1 : e * e * (3 - 2 * e));
/** Soft membership (1 inside, 0 outside, ~0.03 IOD blur) of an ellipse at (a, b) with radii (ra, rb). */
const blob = (u: number, v: number, a: number, b: number, ra: number, rb: number) =>
  1 - smooth((Math.sqrt(((u - a) / ra) ** 2 + ((v - b) / rb) ** 2) - 1) / (0.03 / Math.min(ra, rb)) + 0.5);

/**
 * Colour at canonical point (u, v) and its skin membership (0..1). Features
 * have soft edges and touch the patch borders as on a real face (brows at the
 * glabella / lateral forehead, nostrils under the nasal dorsum, lips under the
 * philtrum); a warm side light gives a coloured (not only luminance) gradient.
 */
function shade(u: number, v: number): [number, number, number, number] {
  const r2 = (u / 1.25) ** 2 + ((v - 0.3) / 1.65) ** 2;
  const face = 1 - smooth((Math.sqrt(r2) - 1) / 0.03 + 0.5);
  if (face <= 0) return [BG[0], BG[1], BG[2], 0];
  const au = Math.abs(u);
  const feats: [number, number[]][] = [
    [blob(au, v, 0.5, 0, 0.18, 0.08), DARK],                     // eyes
    [blob(au, v, 0.42, -0.27, 0.3, 0.045), BROW],                // brows (inner end at |u| ~ 0.12)
    [blob(au, v, 0.1, 0.4, 0.05, 0.035), DARK],                  // nostrils
    [blob(u, v, 0, 1.0, 0.35, 0.08), LIP],                       // lips
  ];
  let col = [...SKIN], skin = 1;
  for (const [m, c] of feats) if (m > 0) { for (let k = 0; k < 3; k++) col[k] = col[k] * (1 - m) + c[k] * m; skin *= 1 - m; }
  const sh = 1 - 0.35 * r2 * r2;
  const tex = 1 + 0.04 * hash2(Math.floor(u * 80), Math.floor(v * 80)); // ~1 px texture at IOD 70-80
  const out = [0, 1, 2].map((k) => (col[k] * sh * tex * (1 + 0.15 * u * WARM[k])) * face + BG[k] * (1 - face));
  return [out[0], out[1], out[2], skin * face];
}

const WARM = [1.3, 1.0, 0.6];

export interface Pose { cx: number; cy: number; iod: number; roll: number }

/**
 * Render the crop [ox, ox+cw) x [oy, oy+ch) of a W x H frame at time t.
 * pulse: relative pulse waveform value (multiplies the skin pulsatility vector).
 */
export function renderCrop(pose: Pose, ox: number, oy: number, cw: number, ch: number, pulse: number, noise: () => number, noiseSd: number): ImageData {
  const data = new Uint8ClampedArray(cw * ch * 4);
  const c = Math.cos(-pose.roll), s = Math.sin(-pose.roll);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const dx = (ox + x + 0.5 - pose.cx) / pose.iod, dy = (oy + y + 0.5 - pose.cy) / pose.iod;
    const u = c * dx - s * dy, v = s * dx + c * dy;
    const [r, g, b, skin] = shade(u, v);
    const p = 4 * (y * cw + x);
    const col = [r, g, b];
    for (let k = 0; k < 3; k++) data[p + k] = col[k] * (1 + skin * pulse * PV[k]) + noiseSd * noise();
    data[p + 3] = 255;
  }
  return { width: cw, height: ch, data, colorSpace: 'srgb' } as ImageData;
}

