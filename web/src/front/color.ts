// color.ts — colour conversions used by pixel selection and the ITA readout.

/** YCbCr as MATLAB rgb2ycbcr on uint8 (BT.601 studio range), same as the seed. */
export function ycbcr(r: number, g: number, b: number): [number, number, number] {
  return [
    16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255,
    128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255,
    128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255,
  ];
}

const lin = (c: number) => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const fLab = (t: number) => (t > (6 / 29) ** 3 ? Math.cbrt(t) : t / (3 * (6 / 29) ** 2) + 4 / 29);

/** sRGB (0..255) -> CIE L*a*b*, D65 white. */
export function srgbToLab(r: number, g: number, b: number): [number, number, number] {
  const R = lin(r), G = lin(g), B = lin(b);
  const X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B;
  const Y = 0.2126729 * R + 0.7151522 * G + 0.072175 * B;
  const Z = 0.0193339 * R + 0.119192 * G + 0.9503041 * B;
  const fx = fLab(X / 0.95047), fy = fLab(Y / 1.0), fz = fLab(Z / 1.08883);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** Individual Typology Angle (README C2): atan2(L* - 50, b*) in degrees. */
export function ita(L: number, bStar: number): number {
  return (Math.atan2(L - 50, bStar) * 180) / Math.PI;
}

/** ITA skin category (Chardon 1991 / Del Bino thresholds). */
export function itaCategory(deg: number): string {
  if (deg > 55) return 'very light';
  if (deg > 41) return 'light';
  if (deg > 28) return 'intermediate';
  if (deg > 10) return 'tan';
  if (deg > -30) return 'brown';
  return 'dark';
}
