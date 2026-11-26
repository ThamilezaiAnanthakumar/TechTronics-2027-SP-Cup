import { describe, expect, it } from 'vitest';
import { ita, srgbToLab } from '../src/front/color';

describe('ITA = atan2(L* - 50, b*) * 180 / pi', () => {
  it('matches hand-computed angles from L*, b*', () => {
    expect(ita(60, 10)).toBeCloseTo(45, 10);
    expect(ita(50, 20)).toBeCloseTo(0, 10);
    expect(ita(40, 10)).toBeCloseTo(-45, 10);
    expect(ita(100, 0)).toBeCloseTo(90, 10);
    expect(ita(50 + Math.sqrt(3) * 10, 10)).toBeCloseTo(60, 10);
  });
  // sRGB -> linear -> XYZ (D65, IEC 61966-2-1 matrix) -> CIELAB, computed
  // independently in Python with the standard formulas.
  const cases: [number[], number, number, number][] = [
    [[255, 255, 255], 100.0, 0.0, 90.0],
    [[224, 172, 138], 74.3112, 24.8331, 44.3916],
    [[141, 85, 36], 41.6711, 37.2391, -12.6073],
    [[90, 56, 37], 27.119, 18.0563, -51.7216],
  ];
  for (const [rgb, L, b, deg] of cases) {
    it(`sRGB ${rgb.join(',')} -> L* ${L}, b* ${b}, ITA ${deg} deg`, () => {
      const lab = srgbToLab(rgb[0], rgb[1], rgb[2]);
      expect(lab[0]).toBeCloseTo(L, 3);
      expect(lab[2]).toBeCloseTo(b, 3);
      expect(ita(lab[0], lab[2])).toBeCloseTo(deg, 3);
    });
  }
});
